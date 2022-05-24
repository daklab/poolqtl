# poolqtl

Code to detect allele specific binding (ASB) of RNA binding proteins (RBP) to RNA from pooled (from multiple individuals) RNA immunoprecipation & sequencing (RIP-seq) data. 

Analysis pipeline is in /src. Underlying code is in the package /src/poolQTL. 

## TODO

- [ ] Compare sQTL enrichment for different models (beta and Gaussian models, Peter's mixture model) and different significance thresholds for calling ASB. 
- [ ] Add more features to sQTL enrichment analysis (e.g. distance to junction/cluster?) 
- [ ] Package poolQTL by making setup.py and specifying requirements. 
