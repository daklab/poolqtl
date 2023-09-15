
import pandas as pd
from sklearn.linear_model import LinearRegression

import numpy as np

import scipy.stats

from . import io_utils, pyro_utils

import torch
import matplotlib.pyplot as plt

torch_matmul = lambda x,y : (torch.tensor(x) @ torch.tensor(y)).numpy() # do we need this? apparently yes!?

def deconvolve(geno, dat, sample_inds = range(5,16), total_thres = 100, plot = True, outfile=None):
    
    # join genotype data and input allele counts
    merged = geno.merge(dat, on = ["variantID", "refAllele", "altAllele"]) # should we also join on contig? 

    # consider different defitions of ref vs alt
    geno_flip = geno.rename(columns={"altAllele" : "refAllele", "refAllele":"altAllele"})
    geno_flip.iloc[:,sample_inds] = 1. - geno_flip.iloc[:,sample_inds]
    merged_flip = geno_flip.merge(dat, on = ["variantID", "refAllele", "altAllele"])
    combined = pd.concat((merged,merged_flip), axis=0) # this handles the misordering of alt/ref correctly
    
    # remove any rows with missigness genotypes
    to_keep = np.isnan(combined.iloc[:,sample_inds]).mean(1) == 0. # keep 96%
    combined = combined[to_keep].copy()

    combined["allelic_ratio"] = combined.altCount / combined.totalCount
    
    # only perform deconv using SNPs with >total_thres total counts
    comb_sub = combined[combined.totalCount >= total_thres].copy()

    X = comb_sub.iloc[:,sample_inds].to_numpy() # dosage matrix
    y = comb_sub.allelic_ratio.to_numpy() # observed allelic proportions

    reg_nnls = LinearRegression(positive=True, fit_intercept=False)
    reg_nnls.fit(X, y)
    w = reg_nnls.coef_
    if plot or outfile is not None:
        fig, (ax3, ax1, ax2) = plt.subplots(3, figsize=(7, 11))
        fig.tight_layout(pad = 4.0)
        #fig.suptitle("sum(w)=%f ideally would be 1" % w.sum())
        combined["pred"] = torch_matmul(combined.iloc[:,sample_inds].to_numpy(), w)
        #combined["pred"] = combined.iloc[:,sample_inds].to_numpy() @  w 

        n_keep = np.sum(combined[combined.totalCount > total_thres].totalCount)
        ax3.hist(combined.totalCount, log=True)
        ax3.axvline(x=total_thres, color='r', linestyle='dashed', linewidth=1)
        ax3.set(xlabel = "# of reads observed with SNP", ylabel = "# of SNPs",
                title = f"{n_keep:,} SNPs with >= {total_thres} reads per SNP")

        ax1.set_title("sum(w)=%f ideally would be 1" % w.sum())
        ax1.bar(x = range(len(w)), height=w*100)
        ax1.set(xlabel="Cell line", ylabel="% representation in sample")

        combined_30 = combined[combined.totalCount >= 30]
        corr,_ = scipy.stats.pearsonr(combined_30.pred, combined_30.allelic_ratio)
        R2 = corr*corr

        ax2.scatter(combined_30.pred, combined_30.allelic_ratio, alpha = 0.05)
        ax2.set_title("R2=%.3f" % R2)
        ax2.set(xlabel="Predicted allelic ratio from genotype", ylabel="Observed allelic ratio in input")
        if outfile is not None:
            fig.savefig(outfile)
        if not plot:
            plt.close(fig)
        

def merge_geno_and_counts(sanger, 
                          dat, 
                          dat_IP, 
                          w, 
                          suffixes = ["_hg19",""],
                          sample_inds = range(5,16),
                          num_haploids = 18,
                          input_total_min = 10, 
                          allele_count_min = 4, 
                          ip_total_min = 30,
                          plot = True):
    """sanger: genotype data
    dat: input alleleic counts
    dat_IP: IP allelic counts
    w: pre-estimated deconvolution betas
    
    Returns
    -------
    merged: merged df with all allelic counts and estimated allelic ratio
    dat_sub: data filtered for sufficient allelic reads to test SNP"""
    
    # We currently join using rsID to get around the mixutre of hg19 and hg38 coords. This causes a memory blow up if there are missing rsIDs (denoted by "."), so filter those ~11% of SNPs out. 
    dat = dat[dat.variantID != "."]
    dat = dat[~dat.variantID.duplicated()]
    dat_IP = dat_IP[dat_IP.variantID != "."]
    dat_IP = dat_IP[~dat_IP.variantID.duplicated()]

    # have to match on rsID because sanger.vcf is hg19 and allelic counts are on hg38
    print("Joining genotype and input allelic counts")
    imp_merged = sanger.rename(columns = {"SNP" : "variantID"}
                              ).merge(dat, 
                                      on = ["contig", "variantID", "refAllele", "altAllele"],
                                     suffixes = suffixes) # sanger is hg19
    # there are only 0.08% flipped alleles so not worth doing.
    # np.isnan(imp_merged.iloc[:,5:16]).any() all False
    imp_merged["input_ratio"] = imp_merged.altCount / imp_merged.totalCount
    X = 0.5 * imp_merged.iloc[:,sample_inds].to_numpy().copy()
    # p = X @ w # WTF doesn't this work!? 
    # p = np.dot(X,w) # doesn't work either
    #p_ = np.array([ X[i] @ w for i in range(X.shape[0]) ])
    imp_merged["pred"] = torch_matmul(X, w)
    
    if plot:
        imp_merged_30 = imp_merged[imp_merged.totalCount >= 30]
        corr,_ = scipy.stats.pearsonr(imp_merged_30.pred, imp_merged_30.input_ratio)
        R2 = corr*corr
        plt.scatter(imp_merged_30.pred, imp_merged_30.input_ratio, alpha = 0.005) 
        plt.title("R2=%.3f" % R2)
        plt.xlabel("Predicted from genotype")
        plt.ylabel("Observed in input")
        plt.show()

    # merge (imp_geno+input) with IP
    print("Joining genotype+input with IP allelic counts")
    merged = imp_merged.drop(labels=sanger.columns[sample_inds], axis=1 # # .rename(columns={"position_y":"position"} # ,"contig_x":"contig" ?
                            ).merge(dat_IP, 
                                    on = ("contig", "position", "variantID", "refAllele", "altAllele"), 
                                    suffixes = ("_input", "_IP"))
    #merged = merged.drop(labels=["contig_y", "position_x" ], axis=1)
    
    merged["IP_ratio"] = merged.altCount_IP / merged.totalCount_IP
    
    dat_sub = merged[merged.totalCount_input >= input_total_min].rename(columns = {"pred" : "pred_ratio"})
    dat_sub = dat_sub[dat_sub.refCount_input >= allele_count_min]
    dat_sub = dat_sub[dat_sub.altCount_input >= allele_count_min]
    dat_sub = dat_sub[dat_sub.totalCount_IP >= ip_total_min]
    dat_sub = dat_sub[dat_sub.pred_ratio >= 0.5/num_haploids]
    dat_sub = dat_sub[dat_sub.pred_ratio <= (1.-0.5/num_haploids)]

    return merged,dat_sub
