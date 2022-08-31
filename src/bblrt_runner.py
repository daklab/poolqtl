import papermill 

to_do = {"HNRNPA1" : 1, "HNRNPK" : 2, "RBFOX2" : 1}

conc = 300

for rbp,n_reps in to_do.items(): 
    for rep in range(n_reps):
        print(rbp,rep)
        papermill.execute_notebook('bblrt_template.ipynb', 
                            'generated_notebooks/bblrt_%s_rep%i.ipynb' % (rbp, rep), 
                            parameters = {"conc" : conc, 
                                         "input_file" : "../results/%s/beta_struct_rep%i.tsv.gz" % (rbp, rep), 
                                         "output_file" : "../results/%s/beta_struct_bblrt_rep%i.tsv.gz" % (rbp, rep)})