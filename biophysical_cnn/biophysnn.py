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

def run_one_epoch(dataloader, 
                  model, 
                  optimizer = None, 
                  regression = False,
                  conc = 30.,
                  binding_threshold = 3.,
                 verbose = False):

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
        if verbose: print("Batch", i, end = "\r")
        
        if regression: 
            input_counts = batch[1]
            IP_counts = batch[2]
            bb = BetaBinomialReparam(output.sigmoid(), 
                                     conc, 
                                     total_count = input_counts + IP_counts,
                                     eps = 1e-10)
            loss = -bb.log_prob(IP_counts).mean()
            enrich = IP_counts / (input_counts + 1.)
            y = enrich > binding_threshold # gives about 10% positive, doubtless an overestimate
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
    aupr = sklearn.metrics.average_precision_score( labels, preds )
    
    accuracy = scipy.stats.pearsonr(preds, enrichs)[0] if regression else np.mean( (preds > 0.) == labels ) 

    return( EpochMetrics( loss = np.mean(losses),
                         acc = accuracy,
                         auroc = auroc,
                        aupr = aupr ), preds, labels, enrichs )
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
    val_metrics_list = []
    train_metrics_list = []

    patience_counter = patience
    best_val_loss = np.inf
    
    for epoch in range(epochs):
        if annealing_flag: 
            if epoch >= len(annealing_schedule): break
            inv_temp = annealing_schedule[epoch]
            model.inverse_temp_unc.data = inv_softplus(inv_temp)
        start_time = timeit.default_timer()
        np.random.seed() # seeds using current time
        train_metrics, _, _, _ = run_one_epoch(train_dataloader, model, optimizer, regression = regression)
        np.random.seed(0) # for consistent randomness in validation
        val_metrics, _, _, _ = run_one_epoch(validation_dataloader, model, optimizer = None, regression = regression)

        val_metrics_list.append(val_metrics)
        train_metrics_list.append(train_metrics)
        
        if val_metrics.loss < best_val_loss: 
            torch.save(model.state_dict(), check_point_filename)
            best_val_loss = val_metrics.loss
            patience_counter = patience
        else: 
            patience_counter -= 1
            if patience_counter <= 0: 
                model.load_state_dict(torch.load(check_point_filename)) # recover the best model so far
                break
        elapsed = float(timeit.default_timer() - start_time)
        if verbose: 
            print("Epoch %i (%.2fs). Train %s. Val %s. Patience: %i" % 
                (epoch+1, 
                 elapsed,
                 " ".join([k+":%.4f" % v for k,v in train_metrics._asdict().items()]),
                 " ".join([k+":%.4f" % v for k,v in val_metrics._asdict().items()]),
                 patience_counter ))

    return val_metrics_list, train_metrics_list


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

