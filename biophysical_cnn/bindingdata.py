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
            right = midpoint + (self.context_length+1)//2
        else: 
            left = start if (width == self.context_length) else np.random.randint(
                start, end - self.context_length)
            right = left + self.context_length
        seq = self.genome[self.chrom[idx]][left:right]
        assert(len(seq)==self.context_length)
        if self.rna and self.strand[idx] == "-": seq = seq_utils.reverse_complement(seq)
        return(seq_utils.one_hot(seq), np.float32(y)) 
    
class IPcountDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 data, 
                 genome, 
                 context_length, 
                 rna = True):
        super().__init__()
        self.context_length = context_length
        self.genome = genome
        self.rna = rna
        
        self.start = data.start.to_numpy()
        self.end = data.end.to_numpy()
        self.chrom = data.chrom.to_numpy()
        self.IP_counts = data.IP_counts.to_numpy()
        self.input_counts = data.input_counts.to_numpy()
        self.strand = data.strand.to_numpy()
        
    def __len__(self):
        return len(self.start)
    
    def __getitem__(self, idx): 
        end = self.end[idx]
        start = self.start[idx]
        width = end - start
        if (width < self.context_length): # always take mid point for positives
            midpoint = int(.5 * (start + end))
            left = midpoint - self.context_length//2
            right = midpoint + self.context_length//2
        else: 
            left = start if (width == self.context_length) else np.random.randint(
                start, end - self.context_length)
            right = left + self.context_length
        seq = self.genome[self.chrom[idx]][left:right]
        if self.rna and self.strand[idx] == "-": seq = seq_utils.reverse_complement(seq)
        return(seq_utils.one_hot(seq), self.input_counts[idx], self.IP_counts[idx] ) 

