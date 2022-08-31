import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pyro.distributions as dist
import pyximport;
pyximport.install(reload_support=True)
import seq_utils
import scipy.stats
import timeit
from bindingdata import FastBedPeaksDataset, IPcountDataset

import sklearn.metrics
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from collections import namedtuple 

from ray import tune

import pickle

EpochMetrics = namedtuple('EpochMetrics', ('loss', 'acc', 'auroc', 'aupr'))

def plot_motifs(pwms):
    pwms_norm = pwms - pwms.mean(1,keepdims=True) 
    nMotif = pwms_norm.shape[0]
    plt.figure(figsize=(4,nMotif))
    for i in range(nMotif): 
        ax = plt.subplot(nMotif,1,i+1)
        plt.axis('off')
        pwm_df = pd.DataFrame(data = pwms_norm[i,:,:].t().numpy(), columns=("A","C","G","T"))
        crp_logo = logomaker.Logo(pwm_df, ax=ax) 


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
    
class Log(nn.Module): 

    def forward(self, x):
        return x.log()  

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
                 pooling = nn.MaxPool1d):
        
        super().__init__()
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
    
class PositiveConv1d(nn.Conv1d):
    """
    Tried this in place of AvgPool1D in LogConvExp net, didn't move much from initialization, and random init didn't work/converge. 
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        # initialize to approximate AvgPool1D
        self.weight.data.fill_(-10.)
        one = torch.expm1(torch.tensor(1./self.weight.size(2))).log() # inverse softplus
        numc = min(self.weight.size(0), self.weight.size(1))
        for i in range(numc): self.weight.data[i,i,:] = one

    def forward(self, x):
        return self._conv_forward(x, F.softplus(self.weight), self.bias)
    
    
class LogConvExpNet(nn.Module):

    def __init__(self, 
                 n_output_channels = 1, 
                 filter_widths = [10, 5], 
                 num_chunks = 5, 
                 pool_width = 4,
                 pool_stride = None, # defaults to pool_width
                 nchannels = [4, 32, 32],
                 n_hidden = 32, 
                 dropout = 0.2, 
                 avgpool = True, # as opposed to maxpool
                 exp_log = True): # conv>exp>pool>log rather than conv>elu>pool
        
        super().__init__()
        
        self.rf = 0 # running estimate of the receptive field
        self.chunk_size = 1 # running estimate of num basepairs corresponding to one position after convolutions
        
        if pool_stride is None: pool_stride = pool_width
        
        n_conv_layers = len(nchannels)-1
        if type(avgpool) == bool: avgpool = [avgpool] * n_conv_layers
        if type(pool_stride) == int: pool_stride = [pool_stride] * n_conv_layers
        if type(pool_width) == int: pool_width = [pool_width] * n_conv_layers
        if type(exp_log) == bool: exp_log = [exp_log] * n_conv_layers
        
        conv_layers = []
        for i in range(len(nchannels)-1):
            conv_layers += [ nn.Conv1d(nchannels[i], 
                                       nchannels[i+1], 
                                       filter_widths[i], 
                                       padding = 0),
                            Exp() if exp_log[i] else nn.ELU(), 
                        (nn.AvgPool1d if avgpool[i] else 
                         nn.MaxPool1d)(pool_width[i], 
                                       stride = pool_stride[i]) ]
            if exp_log[i]: conv_layers.append( Log() )
            assert(filter_widths[i] % 2 == 1) # assume this
            self.rf += (filter_widths[i] - 1) * self.chunk_size
            self.rf += (pool_width[i] - pool_stride[i]) * self.chunk_size # correct? 
            self.chunk_size *= pool_stride[i]

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

def BetaBinomialReparam(mu, conc, total_count, eps = 0.):
    return dist.BetaBinomial(concentration1 = mu * conc + eps, 
                             concentration0 = (1.-mu) * conc + eps, 
                             total_count = total_count)

def run_one_epoch(dataloader, model, optimizer = None, regression = False):

    train_flag = not (optimizer is None)

    torch.set_grad_enabled(train_flag)
    model.train() if train_flag else model.eval() 

    losses = []
    preds = []
    labels = []
    enrichs = []
    
    device = next(model.parameters()).device

    for i,batch in enumerate(dataloader): # collection of tuples with iterator
        
        batch = [ g.to(device) for g in batch ] # transfer data to GPU
        
        output = model(batch[0]) # forward pass. x==batch[0]
        output = output.squeeze() # remove spurious channel dimension if necessary
        
        assert(not output.isnan().any())
        print(i, end = "\r")
        
        if regression: 
            input_counts = batch[1]
            IP_counts = batch[2]
            bb = BetaBinomialReparam(output.sigmoid(), 
                                     30., 
                                     total_count = input_counts + IP_counts,
                                     eps = 1e-10)
            loss = -bb.log_prob(IP_counts).mean()
            enrich = IP_counts / (input_counts + 1.)
            y = enrich > 3. # gives about 10% positive, doubtless an overestimate
            enrichs.append(enrich.detach().cpu().numpy())
        else: 
            y = batch[1]
            loss = F.binary_cross_entropy_with_logits( output, y ) # numerically stable

        if train_flag: 
            loss.backward() # back propagation
            optimizer.step()
            optimizer.zero_grad()

        preds.append(output.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        
        losses.append(loss.detach().cpu().numpy())
    
    preds = np.concatenate(preds)
    enrichs = np.concatenate(enrichs) if regression else None
    labels = np.concatenate(labels)
    auroc = sklearn.metrics.roc_auc_score( labels, preds )

    accuracy = scipy.stats.pearsonr(preds, enrichs)[0] if regression else np.mean( (preds > 0.) == labels ) 

    return( np.mean(losses), accuracy, auroc, preds, labels, enrichs )


def train_model(model, 
                train_data, 
                validation_data, 
                genome,
                regression = False,
                epochs=100, 
                patience=10, 
                verbose = True,
                num_workers = 8,
                annealing_schedule = None, 
                check_point_filename = 'checkpoint.pt', 
                **kwargs): # to save the best model fit to date)
    """
    Train a 1D CNN model and record accuracy metrics.
    """
    # Reload data
    DatasetClass = IPcountDataset if regression else FastBedPeaksDataset
    train_dataset = DatasetClass(train_data, genome, model.seq_len)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=1000, 
                                                   num_workers = num_workers, 
                                                   shuffle = True)
    
    val_dataset = DatasetClass(validation_data, genome, model.seq_len)
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                        num_workers = num_workers, 
                                                        batch_size=1000) # no shuffle important for consistent "random" negatives! 

    # Set up model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    annealing_flag = not annealing_schedule is None
    parameters = [ v for k,v in model.named_parameters() if k != "inverse_temp_unc"
                 ] if annealing_flag else model.parameters()        
    optimizer = torch.optim.Adam(parameters, amsgrad=True, **kwargs)

    # Training loop w/ early stopping
    train_accs = []
    val_accs = []
    train_aucs = []
    val_aucs = []
    
    patience_counter = patience
    best_val_loss = np.inf
    
    for epoch in range(epochs):
        if annealing_flag: 
            if epoch >= len(annealing_schedule): break
            inv_temp = annealing_schedule[epoch]
            model.inverse_temp_unc.data = inv_softplus(inv_temp)
        start_time = timeit.default_timer()
        np.random.seed() # seeds using current time
        train_loss, train_acc, train_auc, _, _, _ = run_one_epoch(train_dataloader, model, optimizer, regression = regression)
        np.random.seed(0) # for consistent randomness in validation
        val_loss, val_acc, val_auc, _, _, _ = run_one_epoch(validation_dataloader, model, optimizer = None, regression = regression)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        if val_loss < best_val_loss: 
            torch.save(model.state_dict(), check_point_filename)
            best_val_loss = val_loss
            patience_counter = patience
        else: 
            patience_counter -= 1
            if patience_counter <= 0: 
                model.load_state_dict(torch.load(check_point_filename)) # recover the best model so far
                break
        elapsed = float(timeit.default_timer() - start_time)
        if verbose: 
            print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f auc %.3f. Val loss: %.4f acc: %.4f auc %.3f. Patience left: %i" % 
            (epoch+1, elapsed, train_loss, train_acc, train_auc, val_loss, val_acc, val_auc, patience_counter ))
    return train_accs, val_accs, train_aucs, val_aucs


def eval_model(model, 
                data, 
                genome,
                regression = False, 
                num_workers = 8, 
                **kwargs): # to save the best model fit to date)
    """
    Evaluate a 1D CNN model and record accuracy metrics.
    """
    DatasetClass = IPcountDataset if regression else FastBedPeaksDataset
    val_dataset = DatasetClass(data, genome, model.seq_len)
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                        num_workers = num_workers, 
                                                        batch_size=1000) # no shuffle important for consistent "random" negatives! 
    # Set up model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    np.random.seed(0) # for consistent randomness in validation
    return run_one_epoch(validation_dataloader, model, optimizer = None, regression = regression)


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
                        phys_net.load_state_dict(torch.load(check_point_filename)) 
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
                    train_accs, val_accs, train_aucs, val_aucs = train_model(phys_net, 
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



class Trainable(tune.Trainable):
    """This is designed to work with ray.tune's hyperparameter tuning (and specifically enables reusing loaded data),
    but can also be used stand-alone by just calling .my_train()"""
    
    def load_data(self):
        """Pre-load all data."""
        
        
        binding_data = pd.read_csv("/gpfs/commons/home/daknowles/RIPnet/all_hnrnpk_rep1_stranded.narrowPeak.gz", sep='\t', usecols=range(6), names=("chrom","start","end","name","score","strand"))
        binding_data = binding_data[ ~binding_data['chrom'].isin(["chrX","chrY"]) ] # only keep autosomes (non sex chromosomes)
        binding_data = binding_data.sort_values(['chrom', 'start']).drop_duplicates() # sort so we can interleave negatives
        binding_data

        unbound = pd.read_csv("/gpfs/commons/home/daknowles/pooledRBPs/biophysical_cnn/unbound_regions.bed.gz", 
                      sep = "\t", 
                      names = ["chrom", "start", "end", "name", "strand", "counts", "length"],
                      index_col = False)
        
        unbound["score"] = 0
        bind_all = pd.concat([binding_data,
                   unbound.loc[:, ~unbound.columns.isin(['counts', 'length'])]], 
                  axis=0)

        test_chromosomes = ["chr1"]
        self.test_data = bind_all[ bind_all['chrom'].isin( test_chromosomes ) ]

        validation_chromosomes = ["chr2","chr3"]
        self.validation_data = bind_all[ bind_all['chrom'].isin(validation_chromosomes) ]

        train_chromosomes = ["chr%i" % i for i in range(4, 22+1)]
        self.train_data = bind_all[ bind_all['chrom'].isin( train_chromosomes ) ]

        self.genome = pickle.load(open("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.pkl","rb")) 

    def reset_config(self, config):
        """Set or reset configuration, including setting up the CNN. """
        
        self.config = config
        
        pool_stride = [config["first_pool_stride"]] + (config["depth"]-1) * [ config["pool_stride"] ]
        
        self.model = LogConvExpNet(
            nchannels = [4] + [config["n_channel"]] * config["depth"], 
            num_chunks = config["num_chunks"],
            filter_widths = [config["first_filter_width"]] + (config["depth"]-1) * [ config["filter_width"] ],
            pool_stride = pool_stride, 
            pool_width = np.array(pool_stride) + config["pool_overlap"], 
            n_hidden = config["n_hidden"],
            dropout = config["dropout"],
            avgpool = [config["avgpool"][0]] + [ config["avgpool"][1] ]*(config["depth"]-1), # don't allow max -> avg
            exp_log = [config["exp_log"][0]] + [ config["exp_log"][1] ]*(config["depth"]-1) # don't allow Relu -> exp
        ).to(self.device)
        
        assert((config["min_seq_len"] <= self.model.seq_len) and (self.model.seq_len <= config["max_seq_len"]) )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                 lr = config["lr"], 
                                 weight_decay = config["weight_decay"],
                                 amsgrad=True)
        
        train_dataset = FastBedPeaksDataset(self.train_data, 
                                            self.genome, 
                                            self.model.seq_len  )
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle = True, num_workers = 0)
        
        validation_dataset = FastBedPeaksDataset(self.validation_data, self.genome, self.model.seq_len)
        self.validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config["batch_size"])

        return True # not sure if we really need to do this but ray.tune docs have it so... 
    
    def setup(self, config): 
        """Initial setup including loading data. Note for ray.tune this maybe running on a different thread or even machine to the master, 
        so the data needs to be loaded here. reset_config allows a new model to be setup however, without reloading data."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device) # keep an eye on whether we're getting GPU or not.
        
        self.load_data()
        
        self.reset_config(config)      

        if str(self.device) == "cpu": 
            print("Setting num threads")
            torch.set_num_threads(28) # needed?


    def run_one_epoch(self, train_flag, dataloader, regression = False):

        torch.set_grad_enabled(train_flag)
        self.model.train() if train_flag else self.model.eval() 

        losses = []
        preds = []
        labels = []
        enrichs = []

        device = next(self.model.parameters()).device

        for i,batch in enumerate(dataloader): # collection of tuples with iterator

            batch = [ g.to(device) for g in batch ] # transfer data to GPU

            output = self.model(batch[0]) # forward pass. x==batch[0]
            output = output.squeeze() # remove spurious channel dimension if necessary

            assert(not output.isnan().any())
            #print(i, end = "\r")

            if regression: 
                input_counts = batch[1]
                IP_counts = batch[2]
                bb = BetaBinomialReparam(output.sigmoid(), 
                                         30., 
                                         total_count = input_counts + IP_counts,
                                         eps = 1e-10)
                loss = -bb.log_prob(IP_counts).mean()
                enrich = IP_counts / (input_counts + 1.)
                y = enrich > 3. # gives about 10% positive, doubtless an overestimate
                enrichs.append(enrich.detach().cpu().numpy())
            else: 
                y = batch[1]
                loss = F.binary_cross_entropy_with_logits( output, y ) # numerically stable

            if train_flag: 
                loss.backward() # back propagation
                self.optimizer.step()
                self.optimizer.zero_grad()

            preds.append(output.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

            losses.append(loss.detach().cpu().numpy())

        preds = np.concatenate(preds)
        enrichs = np.concatenate(enrichs) if regression else None
        labels = np.concatenate(labels)
        auroc = sklearn.metrics.roc_auc_score( labels, preds )
        aupr = sklearn.metrics.average_precision_score( labels, preds )
        
        accuracy = scipy.stats.pearsonr(preds, enrichs)[0] if regression else np.mean( (preds > 0.) == labels ) 

        return( EpochMetrics( loss = np.mean(losses),
                             acc = accuracy,
                             auroc = auroc,
                            aupr = aupr ), preds, labels, enrichs )

    def step(self):
        """Used by ray.tune to save to disk."""
        train_metrics,_,_,_ = self.run_one_epoch(True, self.train_dataloader)
        val_metrics,_,_,_ = self.run_one_epoch(False, self.validation_dataloader)
        return { "val_loss" : val_metrics.loss, 
                 "val_acc" : val_metrics.acc, 
                "val_auroc" : val_metrics.auroc }

    def save_checkpoint(self, checkpoint_dir):
        """Used by ray.tune to save to disk."""
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), path) # and optimizers
        return checkpoint_dir
    
    def load_checkpoint(self, checkpoint_dir):
        """Used by ray.tune to load from disk."""
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def my_train(self, 
              checkpoint_dir = None, 
              verbose = False, 
              patience = 10, 
              max_epochs = 100, 
              **kwargs):
        """ Training loop if not using ray. Important NOT to call this function `train` or ray.tune will use it instead of step! :( 
        Includes early stopping since not using ray.tune's logic."""
        
        patience_counter = patience
        best_val_loss = np.inf
        
        val_metrics_list = []
        train_metrics_list = []
        
        for epoch in range(max_epochs):
            
            start_time = timeit.default_timer()
            
            train_metrics,_,_,_ = self.run_one_epoch(True, self.train_dataloader)
            val_metrics, _,_, _ = self.run_one_epoch(False, self.validation_dataloader)
            
            val_metrics_list.append(val_metrics)
            train_metrics_list.append(train_metrics)
            
            elapsed = float(timeit.default_timer() - start_time)
            if verbose:
                print("Epoch %i (%.2fs). Train %s. Val %s. Patience: %i" % 
                    (epoch+1, 
                     elapsed,
                     " ".join([k+":%.4f" % v for k,v in train_metrics._asdict().items()]),
                     " ".join([k+":%.4f" % v for k,v in val_metrics._asdict().items()]),
                     patience_counter ))

            if val_metrics.loss < best_val_loss: 
                if checkpoint_dir: self.save_checkpoint(checkpoint_dir)
                best_val_loss = val_metrics.loss
                patience_counter = patience
            else: 
                patience_counter -= 1
                if patience_counter <= 0: 
                    if checkpoint_dir: self.load_checkpoint(checkpoint_dir)
                    break
            
        return train_metrics_list, val_metrics_list
