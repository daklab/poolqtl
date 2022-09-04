import torch

import pyximport;
pyximport.install(reload_support=True)
import seq_utils

import numpy as np

def score_variants(phys_net, 
                  snpdata, 
                  genome, 
                  batchsize = 100,
                  output_nonlin = torch.sigmoid ,
                  device = None,
                  verbose = True):
    context_length = phys_net.seq_len
    if device is None: device = "cuda:0" if torch.cuda.is_available() else "cpu"
    phys_net.to(device)
    
    def get_batch(starti, flip_to_alt):
        def get_oh(i): 
            midpoint = snpdata.position.iloc[i]-1
            seq = genome[snpdata.chrom.iloc[i]][ midpoint - context_length//2:midpoint + context_length//2]
            #assert(seq[context_length//2] == snpdata.refAllele.iloc[i])
            if flip_to_alt: 
                seq = seq[:context_length//2] + snpdata.altAllele.iloc[i] + seq[context_length//2+1:]
                #assert(seq[context_length//2] == snpdata.altAllele.iloc[i])
            if snpdata.strand.iloc[i] == "-": seq = seq_utils.reverse_complement(seq)
            return seq_utils.one_hot(seq)
        endi = min(starti+batchsize, len(snpdata))
        batch = [get_oh(i) for i in range(starti, endi)]
        x = np.dstack(batch).transpose([2,0,1])
        return torch.tensor(x, device = device)

    ref_probs = []
    alt_probs = []
    for i in range(0,snpdata.shape[0],batchsize): # range(0,1000,batchsize):
        if verbose: print("%i out of %i" % (i, snpdata.shape[0]), end = "\r")
        x_ref = get_batch(i, False)
        o_ref = output_nonlin(phys_net(x_ref))
        x_alt = get_batch(i, True)
        o_alt = output_nonlin(phys_net(x_alt))
        ref_probs.append(o_ref.detach().cpu().numpy())
        alt_probs.append(o_alt.detach().cpu().numpy())
    ref_probs = np.concatenate(ref_probs)
    alt_probs = np.concatenate(alt_probs)
    
    return(alt_probs, ref_probs)