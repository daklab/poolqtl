import papermill 
import os.path

peaksdir = "/gpfs/commons/home/mschertzer/asb_model/220708_all_ipsc_ip/alignments/macs/"

pwm_sources = ["oRNAment", "K562", "HepG2"]

to_do = {"HNRNPA1" : "all_hnrnpa1_rep1_stranded_noigg.narrowPeak", 
         "HNRNPK" : "all_hnrnpk_stranded_noigg.narrowPeak", 
         "RBFOX2" : "all_rbfox2_rep2_stranded_noigg.narrowPeak"}

for rbp,peak_file in to_do.items(): 
    for pwm_source in pwm_sources:
        print(rbp,pwm_source)
        notebook_out = 'motifscanner_generated_notebooks/%s_%s.ipynb' % (rbp, pwm_source)
        if os.path.isfile(notebook_out): continue
        papermill.execute_notebook('motifscanner_template.ipynb', 
                            notebook_out, 
                            parameters = {"RBP" : rbp, 
                                         "IP_peak_file" : peaksdir + peak_file, 
                                         "pwm_source" : pwm_source,
                                         "unbound_bed_file" : "unbound_regions/%s_unbound.bed.gz" % rbp })

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

reslist = []
for rbp,peak_file in to_do.items(): 
    for pwm_source in pwm_sources:
        res_file  = "biophys_results/test_settings_%s_%s.tsv" % (rbp, pwm_source)
        res = pd.read_csv(res_file, sep="\t", index_col = False)
        res["RBP"] = rbp
        res["pwm_source"] = pwm_source
        reslist.append(res)
        
allres = pd.concat(reslist)
summary = allres.groupby(["RBP","pwm_source"], as_index=False).agg({"val_auc" : "max"})

allres.sort_values("val_auc").groupby(["RBP","pwm_source"]).tail(1).reset_index(drop=True).sort_values(["RBP","pwm_source"])


lse_only = allres[~allres.posmax & ~allres.motifmax]

plt.figure(figsize=(5,4))
sns.lineplot(x="seqlen", y="val_auc", hue = "RBP", style="pwm_source", data=lse_only)
plt.xlabel("Sequence length")
plt.ylabel("Validation AUROC")
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("figs/motifscanner_seqlen.pdf",bbox_inches='tight')

from ray.tune import Analysis
l = []
for rbp,peak_file in to_do.items(): 
    analysis = Analysis("/gpfs/commons/home/daknowles/ray_results/tuneCNN_%s" % rbp)
    df = analysis.dataframe()
    l.append(pd.DataFrame({"RBP":rbp, "pwm_source":"CNN", "val_auc":df.val_auroc.max()}, index=[0]))
    
l.append(s)

d= pd.concat(l)

import seaborn as sns
sns.barplot(data=d, x = "RBP", y = "val_auc", hue="pwm_source")
plt.ylim(0.5,1)
plt.ylabel("Validation AUROC")
plt.xlabel(None)
plt.savefig("auc.pdf")