import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import pyximport;
pyximport.install(reload_support=True)
import seq_utils

import timeit

import sklearn.metrics
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

def plot_motifs(pwms):
    pwms_norm = pwms - pwms.mean(1,keepdims=True) 
    nMotif = pwms_norm.shape[0]
    plt.figure(figsize=(4,nMotif))
    for i in range(nMotif): 
        ax = plt.subplot(nMotif,1,i+1)
        plt.axis('off')
        pwm_df = pd.DataFrame(data = pwms_norm[i,:,:].t().numpy(), columns=("A","C","G","T"))
        crp_logo = logomaker.Logo(pwm_df, ax=ax) 

# positive example: binding of protein onto sequence (ChIP-seq (TF ChIP-seq))
# negative example: the ones that do not overlap with the positive examples 
# for chip-seq data: also shuffling nucleotides can be done to keep the GC content the same as positive example
# because sequencing has biases with GC content and this would be a way to "fix it"
class BedPeaksIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, atac_data, genome, context_length, rna = True):
        super().__init__()
        self.context_length = context_length
        self.atac_data = atac_data
        self.genome = genome
        self.rna = rna

    def __iter__(self): 
        prev_end = 0
        prev_chrom = ""
        for i,row in enumerate(self.atac_data.itertuples()):
            midpoint = int(.5 * (row.start + row.end))
            seq = self.genome[row.chrom][ midpoint - self.context_length//2:midpoint + self.context_length//2]
            if self.rna and row.strand == "-": seq = seq_utils.reverse_complement(seq)
            yield(seq_utils.one_hot(seq), np.float32(1)) # positive example

            if prev_chrom == row.chrom and prev_end < row.start: 
                midpoint = int(.5 * (prev_end + row.start))
                seq = self.genome[row.chrom][ midpoint - self.context_length//2:midpoint + self.context_length//2]
                if self.rna and row.strand == "-": seq = seq_utils.reverse_complement(seq)
                yield(seq_utils.one_hot(seq), np.float32(0)) # negative example midway inbetween peaks, could randomize
            
            prev_chrom = row.chrom
            prev_end = row.end

class BedPeaksDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 data, 
                 genome, 
                 context_length, 
                 rna = True, 
                 score_threshold = 0.):
        super().__init__()
        self.context_length = context_length
        self.data = data
        self.genome = genome
        self.rna = rna
        self.score_threshold = score_threshold
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        y = self.data.score.iloc[idx] > self.score_threshold
        end = self.data.end.iloc[idx]
        start = self.data.start.iloc[idx]
        width = end - start
        if (width < self.context_length) or y: # always take mid point for positives
            midpoint = int(.5 * (start + end))
            left = midpoint - self.context_length//2
            right = midpoint + self.context_length//2
        else: 
            left = start if (width == self.context_length) else np.random.randint(
                start, end - self.context_length)
            right = left + self.context_length
        seq = self.genome[self.data.chrom.iloc[idx]][left:right]
        if self.rna and self.data.strand.iloc[idx] == "-": seq = seq_utils.reverse_complement(seq)
        return(seq_utils.one_hot(seq), np.float32(y)) 
    
class FastBedPeaksDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 data, 
                 genome, 
                 context_length, 
                 rna = True, 
                 score_threshold = 0.):
        super().__init__()
        self.context_length = context_length
        self.genome = genome
        self.rna = rna
        self.score_threshold = score_threshold
        self.start = data.start.to_numpy()
        self.end = data.end.to_numpy()
        self.chrom = data.chrom.to_numpy()
        self.score = data.score.to_numpy()
        self.strand = data.strand.to_numpy()
        
    def __len__(self):
        return len(self.start)
    
    def __getitem__(self, idx): 
        y = self.score[idx] > self.score_threshold
        end = self.end[idx]
        start = self.start[idx]
        width = end - start
        if (width < self.context_length) or y: # always take mid point for positives
            midpoint = int(.5 * (start + end))
            left = midpoint - self.context_length//2
            right = midpoint + self.context_length//2
        else: 
            left = start if (width == self.context_length) else np.random.randint(
                start, end - self.context_length)
            right = left + self.context_length
        seq = self.genome[self.chrom[idx]][left:right]
        if self.rna and self.strand[idx] == "-": seq = seq_utils.reverse_complement(seq)
        return(seq_utils.one_hot(seq), np.float32(y)) 

class ExpandCoupled(nn.Module):
    
    def forward(self, input):
        (batch, vocab, L) = input.shape
        x = input[:, None, :, :-1] * input[:, :, None, 1:]
        return x.flatten(1,2)

class FlippedConv1d(nn.Conv1d):
    def __init__(
        self,
        *args, 
        **kwargs
    ):
        super(FlippedConv1d, self).__init__(*args, **kwargs)
        (n_out, n_in, k) = self.weight.shape
        init = torch.full( (n_out, k), -1., device = self.weight.device, dtype = self.weight.dtype )
        init[:, -1] = 0. # expect less penalty when there is no flipped out nucleotide

        self.bias = nn.Parameter( init.flatten() )

    def forward(self, input):

        (n_out, n_in, k) = self.weight.shape

        #temp = torch.zeros(n_out * k, n_in, k+1, device = self.weight.device, dtype = self.weight.dtype ) 
        temp = torch.zeros(n_out, k, n_in, k+1, device = self.weight.device, dtype = self.weight.dtype )
        for i in range(k-1):
            #temp[i*n_out:(i+1)*n_out, :, 0:(i+1)] = self.weight[:, :, 0:(i+1)]
            #temp[i*n_out:(i+1)*n_out, :, (i+2):] = self.weight[:, :, (i+1):]
            temp[:, i, :, 0:(i+1)] = self.weight[:, :, 0:(i+1)]
            temp[:, i, :, (i+2):] = self.weight[:, :, (i+1):]
        #temp[(k-1)*n_out:k*n_out,:,:-1] = self.weight
        temp[:, k-1, :, :-1] = self.weight
        temp = temp.flatten(0,1)

        net = super(FlippedConv1d, self)._conv_forward(input, temp, self.bias)
        net = net.exp()
        net = net.transpose(1,2)
        net = F.avg_pool1d(net, k)
        return net.transpose(1,2)

class Exp(nn.Module): 

    def forward(self, x):
        return x.exp()    
    
class CNN_1d(nn.Module):

    def __init__(self, 
                 n_output_channels = 1, 
                 filter_widths = [10, 5], 
                 num_chunks = 5, 
                 pool_factor = 4, 
                 nchannels = [4, 32, 32],
                 n_hidden = 32, 
                 dropout = 0.2,
                 use_flipping = True,
                 use_Exp = True,
                 pooling = nn.MaxPool1d):
        
        super(CNN_1d, self).__init__()
        self.rf = 0 # running estimate of the receptive field
        self.chunk_size = 1 # running estimate of num basepairs corresponding to one position after convolutions

        conv_layers = [ (FlippedConv1d if use_flipping else nn.Conv1d)(nchannels[0], nchannels[1], filter_widths[0], padding = 0), 
                        nn.Identity() if use_flipping else Exp(),
                        (nn.AvgPool1d if use_flipping else nn.MaxPool1d)(pool_factor), 
                       ]
        self.rf += (filter_widths[0] - (0 if use_flipping else 1)) * self.chunk_size # note FlippedConv effectively increases the filter_width by 1
        self.chunk_size *= pool_factor
        next_channels = nchannels[1] # * (filter_widths[0] if use_flipping else 1)
        for i in range(1, len(nchannels)-1):
            conv_layers += [ nn.Conv1d(next_channels, nchannels[i+1], filter_widths[i], padding = 0),
                        pooling(pool_factor), 
                        nn.ELU(inplace=True)  ] # popular alternative to ReLU: https://arxiv.org/abs/1511.07289
            assert(filter_widths[i] % 2 == 1) # assume this
            self.rf += (filter_widths[i] - 1) * self.chunk_size
            self.chunk_size *= pool_factor
            next_channels = nchannels[i+1]

        # If you have a model with lots of layers, you can create a list first and 
        # then use the * operator to expand the list into positional arguments, like this:
        self.conv_net = nn.Sequential(*conv_layers)

        self.seq_len = num_chunks * self.chunk_size + self.rf # amount of sequence context required

        print("Receptive field:", self.rf, "Chunk size:", self.chunk_size, "Number chunks:", num_chunks)

        self.dense_net = nn.Sequential( nn.Linear(nchannels[-1] * num_chunks, n_hidden),
                                        nn.Dropout(dropout),
                                        nn.ELU(inplace=True), 
                                        nn.Linear(n_hidden, n_output_channels) )

    def forward(self, x):
        net = self.conv_net(x)
        net = net.view(net.size(0), -1)
        net = self.dense_net(net)
        return(net)
    
def torch_max_values(x, dim): 
    return x.max(dim).values
    
class PhysNet(nn.Module): 

    def __init__(self, 
                 known_pwm, 
                 max_over_positions = False, 
                 max_over_motifs = False, 
                 motif_then_pos = True, 
                 positive_act = F.softplus, 
                 seq_len = 300): 
        """known_pwm: nMotif x 4 x length"""
        super().__init__()
        nMotif, _, k = known_pwm.shape
        self.seq_len = seq_len
        self.nMotif = nMotif
        self.register_buffer("pwm", known_pwm)
        self.motif_scale_unc = nn.Parameter(torch.ones(nMotif)) # ReLU this to get constrained
        self.motif_offset = nn.Parameter(torch.zeros(nMotif))
        self.scale_unc = nn.Parameter(torch.tensor(1.)) 
        self.offset = nn.Parameter(torch.tensor(0.))
        self.positive_act = positive_act
        motif_summarizer = torch_max_values if max_over_motifs else torch.logsumexp
        position_summarizer = torch_max_values if max_over_positions else torch.logsumexp
        
        def summarizer(x): 
            return position_summarizer( motif_summarizer(x, 1), 1) \
                if motif_then_pos else \
                    motif_summarizer( position_summarizer(x, 2), 1)
        self.summarizer = summarizer

    @property
    def motif_scale(self): 
        return self.positive_act(self.motif_scale_unc)

    @property
    def scale(self): 
        return self.positive_act(self.scale_unc)

    def forward(self, x):
        conv_out = F.conv1d(x, self.pwm) # output will be batch x nMotif x length
        conv_lin = conv_out * self.motif_scale[None,:,None] + self.motif_offset[None,:,None]
        #affin = (conv_lin.logsumexp((1,2)) / (self.nMotif * conv_out.shape[2])) \
        #    if self.use_max else \
        #    (conv_lin.logsumexp(1).max(1).values / self.nMotif)

        affin = self.summarizer( conv_lin )

        return affin * self.scale + self.offset
    
    
class FinePhysNet(nn.Module): 
    """Model to FINEtune PWMs, although can also be given randomly initialized PWMs"""

    def __init__(self, 
                 known_pwm,
                 positive_act = F.softplus, 
                 motif_offset = None,
                 scale_unc = torch.tensor(1.), 
                 offset = torch.tensor(0.), 
                 seq_len = 300): 
        """known_pwm: nMotif x 4 x length"""
        super().__init__()
        nMotif, _, k = known_pwm.shape
        self.seq_len = seq_len
        self.nMotif = nMotif
        self.pwm = nn.Parameter(known_pwm)
        self.motif_offset = nn.Parameter(torch.zeros(nMotif) if (motif_offset is None) else motif_offset)
        self.scale_unc = nn.Parameter(scale_unc) 
        self.offset = nn.Parameter(offset)
        self.positive_act = positive_act
    
    @property
    def motif_scale(self): 
        return self.positive_act(self.motif_scale_unc)

    @property
    def scale(self): 
        return self.positive_act(self.scale_unc)

    def forward(self, x):
        conv_out = F.conv1d(x, self.pwm) # output will be batch x nMotif x length
        conv_lin = conv_out + self.motif_offset[None,:,None]

        affin = conv_lin.logsumexp((1,2))

        return affin * self.scale + self.offset

def run_one_epoch(dataloader, cnn_1d, optimizer = None):

    train_flag = not (optimizer is None)

    torch.set_grad_enabled(train_flag)
    cnn_1d.train() if train_flag else cnn_1d.eval() 

    losses = []
    preds = []
    labels = []

    device = next(cnn_1d.parameters()).device

    for (x,y) in dataloader: # collection of tuples with iterator

        (x, y) = ( x.to(device), y.to(device) ) # transfer data to GPU

        output = cnn_1d(x) # forward pass
        output = output.squeeze() # remove spurious channel dimension if necessary
        loss = F.binary_cross_entropy_with_logits( output, y ) # numerically stable

        if train_flag: 
            loss.backward() # back propagation
            optimizer.step()
            optimizer.zero_grad()

        preds.append(output.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        
        losses.append(loss.detach().cpu().numpy())
    
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    auroc = sklearn.metrics.roc_auc_score( labels, preds )

    accuracy = np.mean( (preds > 0.) == labels )

    return( np.mean(losses), accuracy, auroc )


def train_model(cnn_1d, 
                train_data, 
                validation_data, 
                genome,
                epochs=100, 
                patience=10, 
                verbose = True,
                num_workers = 8, 
                check_point_filename = 'cnn_1d_checkpoint.pt', 
                **kwargs): # to save the best model fit to date)
    """
    Train a 1D CNN model and record accuracy metrics.
    """
    # Reload data
    train_dataset = FastBedPeaksDataset(train_data, genome, cnn_1d.seq_len)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=1000, 
                                                   num_workers = num_workers, 
                                                   shuffle = True)
    
    val_dataset = FastBedPeaksDataset(validation_data, genome, cnn_1d.seq_len)
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                        num_workers = num_workers, 
                                                        batch_size=1000) # no shuffle important for consistent "random" negatives! 

    # Set up model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_1d.to(device)
    optimizer = torch.optim.Adam(cnn_1d.parameters(), amsgrad=True, **kwargs)

    # Training loop w/ early stopping
    train_accs = []
    val_accs = []
    train_aucs = []
    val_aucs = []
    
    patience_counter = patience
    best_val_loss = np.inf
    
    for epoch in range(epochs):
        start_time = timeit.default_timer()
        np.random.seed() # seeds using current time
        train_loss, train_acc, train_auc = run_one_epoch(train_dataloader, cnn_1d, optimizer)
        np.random.seed(0) # for consistent randomness in validation
        val_loss, val_acc, val_auc = run_one_epoch(validation_dataloader, cnn_1d, optimizer = None)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        if val_loss < best_val_loss: 
            torch.save(cnn_1d.state_dict(), check_point_filename)
            best_val_loss = val_loss
            patience_counter = patience
        else: 
            patience_counter -= 1
            if patience_counter <= 0: 
                cnn_1d.load_state_dict(torch.load(check_point_filename)) # recover the best model so far
                break
        elapsed = float(timeit.default_timer() - start_time)
        if verbose: 
            print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f auc %.3f. Val loss: %.4f acc: %.4f auc %.3f. Patience left: %i" % 
            (epoch+1, elapsed, train_loss, train_acc, train_auc, val_loss, val_acc, val_auc, patience_counter ))
    return cnn_1d, train_accs, val_accs, train_aucs, val_aucs


def eval_model(cnn_1d, 
                data, 
                genome,
                num_workers = 8, 
                check_point_filename = 'cnn_1d_checkpoint.pt', 
                **kwargs): # to save the best model fit to date)
    """
    Evaluate a 1D CNN model and record accuracy metrics.
    """
    cnn_1d.load_state_dict(torch.load(check_point_filename)) 
    val_dataset = FastBedPeaksDataset(data, genome, cnn_1d.seq_len)
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                        num_workers = num_workers, 
                                                        batch_size=1000) # no shuffle important for consistent "random" negatives! 
    # Set up model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_1d.to(device)

    np.random.seed(0) # for consistent randomness in validation
    return run_one_epoch(validation_dataloader, cnn_1d, optimizer = None)


def test_settings(specific_pwms, 
                  train_data, 
                  validation_data,
                  genome, 
                  seq_len_range = range(100, 701, 100), 
                 base_checkpoint_dir = "checkpoints/"
                 ): 
    results = []
    
    os.makedirs(base_checkpoint_dir, exist_ok = True)

    for max_over_positions in (False,True): # rather than logSumExp
        for max_over_motifs in (False,True): # rather than logSumExp
            for motif_then_pos in (False,True): # summarize over motif before position
                for seq_len in seq_len_range: 
                    if (not motif_then_pos) and (max_over_positions == max_over_motifs): continue # otherwise is equivalent
                    phys_net = PhysNet(specific_pwms, 
                                        max_over_positions = max_over_positions,
                                         max_over_motifs = max_over_motifs, 
                                         motif_then_pos = motif_then_pos, 
                                         seq_len = seq_len)
                    check_point_filename = base_checkpoint_dir + ("posmax%i_motifmax%i_len%i%s.pt" % (max_over_positions, 
                                                                                           max_over_motifs, 
                                                                                           seq_len,
                                                                                          "" if motif_then_pos else "_posthenmax"))
                    print(check_point_filename)
                    if os.path.isfile(check_point_filename): 
                        _, _, val_aucs = eval_model(phys_net, 
                                              validation_data,
                                              genome, 
                                              check_point_filename = check_point_filename)
                        results.append((max_over_positions, 
                                    max_over_motifs, 
                                    seq_len, 
                                    check_point_filename, 
                                         motif_then_pos,
                                    val_aucs,
                                    0.))
                        continue
                    phys_net, train_accs, val_accs, train_aucs, val_aucs = train_model(phys_net, 
                                                                               train_data, 
                                                                               validation_data, 
                                                                               genome, 
                                                                               verbose = False, 
                                                                               check_point_filename = check_point_filename,
                                                                               lr = 0.1) 
                    torch.save(phys_net.state_dict(), check_point_filename) 
                    results.append((max_over_positions, 
                                    max_over_motifs, 
                                    seq_len, 
                                    check_point_filename, 
                                     motif_then_pos, 
                                   np.max(val_aucs),
                                   np.max(train_aucs)))
                    
    return pd.DataFrame(results, columns = ["posmax", "motifmax", "seqlen", "file", "motif_then_pos", "val_auc", "train_auc"])
