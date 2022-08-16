import torch
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

@dataclass
class ASBdata: 
    snps: pd.Index
    total_count: torch.Tensor
    alt_count: torch.Tensor
    pred_ratio: torch.Tensor
    device: torch.device
    
    @staticmethod
    def from_pandas(df, device = "cpu"): 
        return ASBdata(
            snps = df.variantID,
            total_count = torch.tensor(df.totalCount.to_numpy(), dtype = torch.float, device = device),
            alt_count = torch.tensor(df.altCount.to_numpy(), dtype = torch.float, device = device),
            pred_ratio = torch.tensor(df.pred_ratio.to_numpy(), dtype = torch.float, device = device),
            device = device
        )

    @property
    def num_snps(self):
        return len(self.alt_count)
    
    @property
    def snp_indices(self):
        return torch.arange(self.num_snps, device = self.total_count.device) 
    
@dataclass
class RelativeASBdata: 
    snps: pd.Index
    input_total_count: torch.Tensor
    input_alt_count: torch.Tensor
    IP_total_count: torch.Tensor
    IP_alt_count: torch.Tensor
    pred_ratio: torch.Tensor
    device: torch.device
    
    @staticmethod
    def from_pandas(df, device = "cpu"): 
        return RelativeASBdata(
            snps = df.variantID,
            input_total_count = torch.tensor(df.totalCount_input.to_numpy(), dtype = torch.float, device = device),
            input_alt_count = torch.tensor(df.altCount_input.to_numpy(), dtype = torch.float, device = device),
            IP_total_count = torch.tensor(df.totalCount_IP.to_numpy(), dtype = torch.float, device = device),
            IP_alt_count = torch.tensor(df.altCount_IP.to_numpy(), dtype = torch.float, device = device),
            pred_ratio = torch.tensor(df.pred_ratio.to_numpy(), dtype = torch.float, device = device),
            device = device
        )

    @property
    def num_snps(self):
        return len(self.input_alt_count)
    
    @property
    def snp_indices(self):
        return torch.arange(self.num_snps, device = self.input_total_count.device) # more consistent with rest of class to use torch not np

@dataclass
class ReplicateASBdata: 
    snp_indices: torch.Tensor
    snps: pd.Index
    input_total_count: torch.Tensor
    input_alt_count: torch.Tensor
    IP_total_count: torch.Tensor
    IP_alt_count: torch.Tensor
    pred_ratio: torch.Tensor
    device: torch.device
    
    @staticmethod
    def from_pandas(df, device = "cpu"): 
        
        snp_indices, snps = pd.factorize(df.variantID)
        
        return ReplicateASBdata(
            snp_indices = torch.tensor(snp_indices, dtype = torch.long, device = device),
            snps = snps,
            input_total_count = torch.tensor(df.totalCount_input.to_numpy(), dtype = torch.float, device = device),
            input_alt_count = torch.tensor(df.altCount_input.to_numpy(), dtype = torch.float, device = device),
            IP_total_count = torch.tensor(df.totalCount_IP.to_numpy(), dtype = torch.float, device = device),
            IP_alt_count = torch.tensor(df.altCount_IP.to_numpy(), dtype = torch.float, device = device),
            pred_ratio = torch.tensor(df.pred_ratio.to_numpy(), dtype = torch.float, device = device),
            device = device
        )

    @property
    def num_snps(self):
        return len(self.snps)
    
    @property
    def num_measurements(self):
        return len(self.input_alt_count)


