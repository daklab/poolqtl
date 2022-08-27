import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import pyximport;
pyximport.install(reload_support=True)
import seq_utils

import timeit

from biophysnn import train_model, eval_model
from bindingdata import FastBedPeaksDataset, IPcountDataset

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
    
def inv_softplus(x):
    return torch.expm1(x).log()
    
class FinePhysNet(nn.Module): 
    """Model to FINEtune PWMs, although can also be given randomly initialized PWMs"""

    def __init__(self, 
                 known_pwm,
                 positive_act = F.softplus, 
                 motif_offset = None,
                 scale_unc = torch.tensor(1.), 
                 offset = torch.tensor(0.), 
                 annealing = False, 
                 annealing_init = 0.01, 
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
        annealing_init_unc = inv_softplus(torch.tensor(annealing_init)) # inv_softplus
        self.inverse_temp_unc = nn.Parameter(annealing_init_unc) if annealing else None
    
    @property
    def inverse_temp(self): 
        return F.softplus(self.inverse_temp_unc) if self.inverse_temp_unc else 1. 
    
    @property
    def motif_scale(self): 
        return self.positive_act(self.motif_scale_unc)

    @property
    def scale(self): 
        return self.positive_act(self.scale_unc)

    def forward(self, x):
        conv_out = F.conv1d(x, self.pwm) # output will be batch x nMotif x length
        conv_lin = conv_out + self.motif_offset[None,:,None]
        conv_lin_temp = conv_lin * self.inverse_temp
        
        affin = conv_lin_temp.logsumexp((1,2))
        affin_temp = affin / self.inverse_temp

        return affin_temp * self.scale + self.offset


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
