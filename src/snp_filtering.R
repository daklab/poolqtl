# mostly for Dan: Megan and i met yesterday. one idea we discussed is trying to
# filter for SNPs that we are confident show balance in the input,
#rather than trying to account for ASE in the ASB model.
# the question is how many SNPs we’d even get to test, especially in intronic 
# regions where the input counts are low.
# so, i think a good thing for you to try would be get a binomial confidence 
# interval for the allelic proportion for each input SNP
# i think you can do this with binomial.test
# (actually i think the simulation code probably does that?)
# we’d want that CI to be contained in [p-epsilon, p+epsilon] where p is the expected proportion 
# (i.e. 0.5 in the single individual setting, whatever comes out of the 
# deconvolution regression for the pooled setting)
# you could vary epsilon in some range (maybe something like 0.1 would be a good starting point?)
# and see how many intronic and exonic SNP we would be able to test
# probably easiest with a single individual first: i’m not sure quite how we 
# should handle the situation where p is close to 0 or 1

library(tidyverse)
require(ashr)
require(doMC)
library(data.table)
#rm(ash_result, dat, deconv, deconv_res, fit, geno, geno_sub, here, input, joined, joined_sub, mat, obs, obs_sub, peaks, res_, res1, res2, res3, res4, results, sim_dat, sum_stats, X)

vcf_file <- "/home/dmeyer/projects/bqtls/sanger.vcf.gz"
vcf_file <- "~/projects/bqtls/CIRMlines_flipped.vcf.gz"
vcf_samples <- system(paste("bcftools query -l", vcf_file), intern=T)
sample_index <- "CW70142-1-1194512527_CW70142-1-1194512527" # Expected to be this
#sample_index <- "CW30274-1-1194512543_CW30274-1-1194512543"
vcf_cmd <- paste0("/bin/bcftools query -s '", sample_index, "' -f '%CHROM %POS %ID %REF %ALT [%GT]\n' ",vcf_file)
#"/bin/bcftools query -f '%CHROM %POS %ID %REF %ALT [%GT]\n' ",vcf_file)
vcf <- fread(cmd=vcf_cmd, header = F, sep=" ", col.names = c("chrom", "position", "variantID", "refAllele", "altAllele", "gt"))
gc()
vcf$refCount <- 0
vcf$altCount <- sapply(vcf$gt,
                       function(x) {
                       switch(x,
                              "0|0"=0,
                              "0|1"=1,
                              "1|0"=1,
                              "1|1"=2,
                              "0/0"=0,
                              "0/1"=1,
                              "1/0"=1,
                              "1/1"=2,
                              0)
                       })
vcf$refCount <- 2 - vcf$altCount


input = read_tsv("/home/dmeyer/projects/bqtls/tdp43/allelic/tdp43_input_annotated_allelic.out")%>%rename(chrom=1)
input = input[(input$variantID != ".") & (input$variantID %in% vcf[vcf$altCount == 1,]$variantID),]
input$in_intron = as.numeric((input$in_gene == 1) & (input$in_exon == 0))
input_original = input
#input_idx = (input$totalCount >= 10) & (input$altCount >= 2) & (input$refCount >= 2)
input_idx = (input_original$totalCount >= 10)
# filtering the input to make sure 10 total reads, 2 from each allele
input = input_original[input_idx,]

print(ggplot(data.table(ci=input$altCount/input$totalCount), aes(x = ci))+
    geom_histogram() + 
    labs( x = "Alt allele frequency in input", title = sample_index))

print(ggplot(data.table(x=input$altCount/input$totalCount, y= input$totalCount), 
             aes(x = x,y=y))+
    geom_point(alpha=0.25) + 
    labs( x = "Alt allele frequency in input", y = "Total count", title = sample_index))

#input%>%filter(altCount/totalCount == 0, totalCount > 900)%>%arrange(desc(totalCount))

sum_stats = foreach(i = 1:nrow(input), .combine = bind_rows) %do% {
    here = input[i,]
    
    bt = binom.test(c(here$altCount, here$refCount), p = 0.5)
    lor_ci = log(bt$conf.int)
    or_ci = bt$conf.int
    tibble( 
            ci1 = or_ci[1],
            ci2 = or_ci[2],
            or_mean = mean(bt$conf.int),
            or_se = (or_ci[2] - or_ci[1])/4,
            lor_mean = mean(lor_ci),     
            lor_se = (lor_ci[2] - lor_ci[1])/4 )
}
#sum_stats = sum_stats[input_idx,]
# TODO:
# check that ci[1] <= 0.5 <= c[2] AND that ci[1] and ci[2] both within window of p- epsilon , p+epsilon
# THEN check that 
# filter the input to make sure 10 total reads, 2 from each allele
# rename this file to "/home/dmeyer/projects/bqtls/sanger.vcf.gz"
# CIRM_imputed_genotypes_sanger.vcf.gz
## Check what david did in 1_detect_ASB* for filtering the input/IP data
# Then we can dive into this all together next week

# If you do the filtering for a total count of 10 in the input do the 0s go away?
# David is a bit worried about there being so many zeros
# In the imputation results there is a computationally estimated confidence, and it looks good generally when
# we've matched to sequencing data

# Try putting in the wrong cell line and seeing how the plot differs (do for every cell line maybe?)

# Once you've done this filtering then run the beta model again

# How many are testable? How many does the model think are significant
dev.off()
pdf("2024-02-08_binomial_test.pdf")
{
epsilon = 0.2
p = 0.5
ci = sum_stats$or_mean
idx = apply(sum_stats, 1, function(x) {
    (x['ci1'] >= p - epsilon) &
        (x['ci1'] <= x['ci2']) & 
        (x['ci2'] <= p + epsilon) &
        (x['ci1'] <= 0.5) & (0.5 <= x['ci2'])
})

print(ggplot(data.table(ci, p, epsilon), aes(x = ci))+
    geom_histogram()+labs(title="Epsilon = 0.2 Confidence interval filtering "))
    #labs( x = "Alt allele frequency in input"))
    #geom_vline(xintercept = p - epsilon, lty='dashed', col='red') +
    #geom_vline(xintercept = p + epsilon, lty='dashed', col='red'))

cat("Total SNPs included to test: ", sum(idx),"\n")
cat("Intronic SNPs included to test: ", sum(idx & input$in_intron),"\n")
cat(  "Exonic SNPs included to test: ", sum(idx & input$in_exon), "\n")
cat("Percent of intronic SNPs retained: ",  (sum(idx & input$in_intron)/sum(input$in_intron)*100)%>%round(3),"%\n")
cat("Percent of exonic SNPs retained: ",    (sum(idx & input$in_exon)/sum(input$in_exon)*100)%>%round(3),"%\n")
cat("Intronic SNPs in peaks included to test: ", sum(idx & input$in_intron& input$in_peak),"\n")
cat(  "Exonic SNPs in peaks included to test: ", sum(idx & input$in_exon  & input$in_peak), "\n")
cat("Intronic SNPs in peaks included to test: ", (sum(idx & input$in_intron& input$in_peak)/sum(input$in_intron& input$in_peak)*100)%>%round(3),"%\n")
cat(  "Exonic SNPs in peaks included to test: ", (sum(idx & input$in_exon  & input$in_peak)/sum(input$in_exon  & input$in_peak)*100)%>%round(3), "%\n")
}

write_tsv(sum_stats, "sum_stats.txt")
sum_stats <- read_tsv("sum_stats.txt")
sum_stats <- sum_stats[input_idx,]

if (! all(sum_stats$ci1 <= sum_stats$ci2))
    stop("FIX THIS. Not all ci1 <= ci2")

filter_snps_summary=function(epsilon = 0.1, p=0.5) {
    idx = apply(sum_stats, 1, function(x) {
        (x['ci1'] >= p - epsilon) &
            (x['ci1'] <= x['ci2']) & 
            (x['ci2'] <= p + epsilon) &
            (x['ci1'] <= 0.5) & (0.5 <= x['ci2'])
    })
    data.table(intronic_snps = sum(idx & input$in_intron),
               exonic_snps = sum(idx & input$in_exon),
               epsilon = epsilon, 
               p = p)
}

lapply(c(0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.04), filter_snps_summary)%>%
    rbindlist%>%
    melt(id.vars = c('p', 'epsilon'))%>%
    mutate(epsilon = as.factor(epsilon))%>%
    ggplot(aes(x = epsilon, y = value, fill = variable))+
    geom_bar(stat='identity', position='dodge')+
    labs(x = "epsilon", y = "Number of SNPs", title = "Number of SNPs after thresholding based on binomial test confidence interval")+
    theme_linedraw()+
    scale_y_continuous(trans = "log10")
dev.off()

epsilon <- 0.2
p <- 0.5
idx = apply(sum_stats, 1, function(x) {
        (x['ci1'] >= p - epsilon) &
            (x['ci1'] <= x['ci2']) & 
            (x['ci2'] <= p + epsilon) &
            (x['ci1'] <= 0.5) & (0.5 <= x['ci2'])
    })
sum_stats[idx,]
write_tsv(input[idx,], "/home/dmeyer/projects/bqtls/tdp43/tdp43_filtered_input.txt")