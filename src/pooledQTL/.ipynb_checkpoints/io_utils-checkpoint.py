
import gzip
import numpy as np
import pandas as pd

def smart_open(filename, *args, **kwargs):
    return gzip.open(filename, *args, **kwargs) if filename[-2:]=="gz" else open(filename, *args, **kwargs)

def loadGenotypes(
    genotypeFile, 
    posterior = True, # if true then directly extract dosage. if false convert genotype (GT) to dosage. 
    posterior_index = 1, # which element of the data should we extract as the dosage? 
    maxlines = None, 
    get_confidence = False, 
    add_chr_prefix = "chr",
    print_every = 0
):
    # For now, will assume './.' corresponds to major allele
    genotype_doses = {
        '0|0': 0.0,
        '0|1': 0.5,
        '1|0': 0.5,
        '1|1': 1.0,
        '0/0': 0.0,
        '0/1': 0.5,
        '1/0': 0.5,
        '1/1': 1.0,
        './.': np.nan,
        '.|.': np.nan
    }
    genotype_arr = []
    
    positions = []
    contigs = []
    confidences = []
    snps = []
    alleles_1 = []
    alleles_2 = []
    
    first_line = True
    
    with smart_open(genotypeFile, 'r') as genotypes:
        for idx,line in enumerate(genotypes): 
            if type(line) is bytes: line = line.decode()
            if line[:2] == "##": continue # skip header
            elems = line.strip('\n').split('\t')
            if first_line:
                sample_names = elems[9:]
                first_line = False
                continue
            contig = add_chr_prefix + elems[0]
            pos = int(elems[1])
            snp = str(elems[2]) # use as future key
            #key = (contig, pos)
            allele_1 = str(elems[3])
            allele_2 = str(elems[4])
            if not posterior:
                genotype_list = [genotype_doses[allele.split(':')[0]] for allele in elems[9:]] # map maximum likelihood genotype
            else:
                genotype_list = [ float(allele.split(':')[posterior_index]) for allele in elems[9:] ] # second element is dosage
            if get_confidence:
                confidence = [ max([ float(gp) for gp in allele.split(':')[2].split(",") ]) for allele in elems[9:] ]
                confidences.append(confidence)
            genotype_arr.append(genotype_list)
            
            # Entries below for pandas array
            positions.append(pos)
            contigs.append(contig)
            snps.append(snp)
            alleles_1.append(allele_1)
            alleles_2.append(allele_2)
            
            if print_every and (idx % print_every)==0: print("Processed %d" % idx, end = "\r")
            
            if maxlines and len(snps)>=maxlines: break
    
    df_genotypes_1 = pd.DataFrame(data={
        'position':positions, 
        'contig':contigs, 
        'SNP':snps,
        'refAllele': alleles_1,
        'altAllele': alleles_2}, 
        index=snps)
    df_genotypes_2 = pd.DataFrame(data=np.array(genotype_arr), index=snps, columns=sample_names)
    df_genotypes = pd.concat([df_genotypes_1, df_genotypes_2], axis=1)
    if get_confidence:
        confidences = pd.DataFrame(data=np.array(confidences), index=snps, columns=sample_names)
        return df_genotypes, confidences
    else: return df_genotypes
    
