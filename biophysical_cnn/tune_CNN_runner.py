import papermill 

to_do = {"HNRNPA1" : "_rep1", 
         "HNRNPK" : "", 
         "RBFOX2" : "_rep2"}

for rbp,rep in to_do.items(): 
    print(rbp)
    papermill.execute_notebook('tune_CNNs.ipynb', 
                        'tune_CNN_notebooks/%s_CNN.ipynb' % rbp, 
                        parameters = {"RBP" : rbp,  "rep" : rep})