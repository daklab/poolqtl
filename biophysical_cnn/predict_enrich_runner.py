import papermill 
import os.path

pwm_sources = ["oRNAment", "K562", "HepG2"]

to_do = {"HNRNPA1" : "rep1", 
         "HNRNPK" : "both", 
         "RBFOX2" : "rep2"}

for rbp,rep in to_do.items(): 
    for pwm_source in pwm_sources:
        print(rbp,pwm_source)
        notebook_out = 'predict_enrich_notebooks/%s_%s.ipynb' % (rbp, pwm_source)
        if os.path.isfile(notebook_out): continue
        papermill.execute_notebook('predict_enrichment.ipynb', 
                            notebook_out, 
                            parameters = {"RBP" : rbp, 
                                         "rep" : rep, 
                                         "pwm_source" : pwm_source })
        
