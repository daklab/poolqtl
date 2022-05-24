
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
from sklearn.linear_model import LinearRegression

import numpy as np

from importlib import reload

from pathlib import Path

from pooledQTL import interval_utils

finemap_dir = Path("/gpfs/commons/home/mschertzer/ipsc_sqtl/fine_mapped")
    
max_pips = pd.concat([ 
    pd.read_csv(finemap_dir / ("ipsc_sqtl_finemapped_L10_chr%i.txt.gz" % i), 
                sep = "\t",  
                index_col = False).groupby(["rsid","chromosome","position"],
                                           as_index=False).agg({'pip': 'max'}) for i in range(1,23)], axis=0)

max_pips.rename(columns = {"chromosome" : "chrom", "position" : "position_hg19"}, inplace = True)
max_pips.chrom = 'chr' + max_pips.chrom.astype(str)

dbsnp151 = pd.read_csv("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/snp151Common.txt.gz", 
                       sep = "\t",  
                       index_col = False, 
                       usecols = [1,3,4,15,24], 
                       names = ["chrom","position","rsid","function","maf"])

dbsnp151.maf = dbsnp151.maf.str.split(",", expand=True, n = 2).iloc[:,:2].min(1) # only works for biallelic
plt.hist(dbsnp151.maf[dbsnp151.maf<0.05],100)

max_pips = max_pips.merge(dbsnp151, on = ["chrom", "rsid"])


res =  pd.read_csv("/gpfs/commons/home/daknowles/pooledRBPs/results/rep_struct_asb.tsv.gz", 
                       sep = "\t",  
                       index_col = False)


max_pips["in_peak"] = get_overlap(both_strands, max_pips)
max_pips.in_peak[max_pips.in_peak > 1] = 1 # don't count being in two peaks (e.g. on both strands)

tab = pd.crosstab(max_pips.in_peak, max_pips.pip > 0.1)
print(tab)
scipy.stats.chi2_contingency(tab) # p=0.
scipy.stats.fisher_exact(tab) # 2.09, 4.1e-261

#overlap = res.rename(columns={"variantID":"rsid", "contig" : "chrom"}).merge(max_pips, how = "right", on = ["rsid", "position_hg19", "position", "chrom"])

overlap = res.rename(columns={"variantID":"rsid", "contig" : "chrom"}).merge(max_pips, how = "outer", on = ["rsid", "position_hg19", "position", "chrom"])

overlap["clean_q"] = overlap.q.copy()
overlap.loc[np.isnan(overlap.q),"clean_q"] = 1. 

peaks = { k:pd.read_csv("/gpfs/commons/home/mschertzer/asb_model/all_hnrnpk_rep1_%s_peaks.narrowPeak" % k, 
                        sep = "\t",  
                        index_col = False, 
                       names = ["chrom","start","end","name","score","strand","a","b","c","d"]) for k in ["neg","pos"] }

plt.hist(peaks["neg"].score,100); plt.show()
plt.hist(peaks["pos"].score,100); plt.show()

chroms = [ "chr%i" % i for i in range(1,23) ]
its = {k:interval_utils.to_interval_trees(v, chroms) for k,v in peaks.items() }

both_strands = { chrom : its["pos"][chrom] | its["neg"][chrom] for chrom in chroms }

overlap["in_peak"] = interval_utils.get_overlap(both_strands, overlap)

exons = pd.read_csv("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/gencode.v38.exons.txt.gz", 
                       sep = "\t",  
                       index_col = False, usecols = range(3)).rename(columns = {"chr" : "chrom"})
exons = exons[(exons.end - exons.start) >= 9] # remove super short exons
exons_tree = interval_utils.to_interval_trees(exons, chroms)

overlap["exonic"] = interval_utils.get_overlap(exons_tree, overlap)

genes = pd.read_csv("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/genes.tsv.gz", 
                       sep = "\t",  
                       index_col = False, usecols = range(3)).rename(columns = {"chr" : "chrom"})
genes = genes[genes.end - genes.start >= 100]
genes_tree = interval_utils.to_interval_trees(genes, chroms)

overlap["genic"] = interval_utils.get_overlap(genes_tree, overlap)

overlap.to_csv("/gpfs/commons/home/daknowles/pooledRBPs/results/overlap.tsv.gz", sep = "\t", index = False)

import statsmodels.api as sm

overlap["asb"] = overlap.clean_q < 0.1
X = overlap.loc[:,["asb","in_peak","exonic","genic"]].to_numpy(dtype=float)
log_reg = sm.Logit(overlap.pip > 0.1, X).fit()

