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
        papermill.execute_notebook('Biophysical_CNNs.ipynb', 
                            notebook_out, 
                            parameters = {"RBP" : rbp, 
                                         "IP_peak_file" : peaksdir + peak_file, 
                                         "pwm_source" : pwm_source,
                                         "unbound_bed_file" : "unbound_regions/%s_unbound.bed.gz" % rbp })