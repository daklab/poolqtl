import intervaltree

def to_interval_tree(one_chrom):
    return intervaltree.IntervalTree.from_tuples(list(zip( *[one_chrom.start, one_chrom.end] )))

def to_interval_trees(peakdf, chroms):
    return { chrom: to_interval_tree(peakdf[peakdf.chrom == chrom])  for chrom in chroms }

def get_overlap(peaks, snps): 
    return [ min(len(peaks[row.chrom][row.position]),1) for row in snps.itertuples() ]
    
def get_overlap_window(peaks, snps, window=100000): 
    return [min(len(peaks[row.chrom][row.position-window] | peaks[row.chrom][row.position] | peaks[row.chrom][row.position+window]), 1) for row in snps.itertuples() ]
    