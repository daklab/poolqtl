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

import biophysnn

class Trainable(tune.Trainable):
    """This is designed to work with ray.tune's hyperparameter tuning (and specifically enables reusing loaded data),
    but can also be used stand-alone by just calling .my_train()"""
    
    def load_data(self, config):
        """Pre-load all data."""
        
        basedir = "/gpfs/commons/home/mschertzer/asb_model/220708_all_ipsc_ip/alignments/macs/"
        peak_file = basedir + ("all_%s%s_stranded_noigg.narrowPeak" % (config["RBP"].lower(), config["rep"]))
        binding_data = pd.read_csv(peak_file, sep='\t', usecols=range(6), names=("chrom","start","end","name","score","strand"))
        binding_data = binding_data[ ~binding_data['chrom'].isin(["chrX","chrY"]) ] # only keep autosomes (non sex chromosomes)
        binding_data = binding_data.sort_values(['chrom', 'start']).drop_duplicates() # sort so we can interleave negatives

        unbound = pd.read_csv("/gpfs/commons/home/daknowles/pooledRBPs/biophysical_cnn/unbound_regions/%s_unbound.bed.gz" % config["RBP"], 
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
        
        if hasattr(self, 'config'): assert(self.config["RBP"] == config["RBP"]) # can't change data! 

        self.config = config
        
        pool_stride = [config["first_pool_stride"]] + (config["depth"]-1) * [ config["pool_stride"] ]
        
        self.model = biophysnn.LogConvExpNet(
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
                    
        self.load_data(config)
        
        self.reset_config(config)      

        if str(self.device) == "cpu": 
            print("Setting num threads")
            torch.set_num_threads(28) # needed?


    def run_one_epoch(self, train_flag, dataloader, regression = False):

        return biophysnn.run_one_epoch(dataloader, 
                               self.model, 
                               self.optimizer if train_flag else None, 
                               regression = regression, 
                               conc = self.config["conc"],
                               binding_threshold = self.config["binding_threshold"]
                               )

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

    ### Could delete my_train, only useful for debugging
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
