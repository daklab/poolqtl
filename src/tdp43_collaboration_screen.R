library(tidyverse)
require(ashr)
require(doMC)
library(data.table)
library(feather)


vcf_file <- "/gpfs/commons/home/dmeyer/bindingQTL_share/genotype/chrAll_QCFinished_full_2sample.anno.vcf.gz"
input_files <- c('/gpfs/commons/home/dmeyer/bindingQTL_share/24a-hNIL-control-tdp/allelic/24a-hNIL-control-tdp_input_allelic.out',
    '/gpfs/commons/home/dmeyer/bindingQTL_share/24a-hNIL-c9-tdp/allelic/24a-hNIL-c9-tdp_input_allelic.out',
    '/gpfs/commons/home/dmeyer/bindingQTL_share/24a-hNIP-control-tdp/allelic/24a-hNIP-control-tdp_input_allelic.out',
    '/gpfs/commons/home/dmeyer/bindingQTL_share/24a-hNIP-c9-tdp/allelic/24a-hNIP-c9-tdp_input_allelic.out')
ip_files <- c('/gpfs/commons/home/dmeyer/bindingQTL_share/24a-hNIL-control-tdp/allelic/24a-hNIL-control-tdp_ip_allelic.out',
               '/gpfs/commons/home/dmeyer/bindingQTL_share/24a-hNIL-c9-tdp/allelic/24a-hNIL-c9-tdp_ip_allelic.out',
               '/gpfs/commons/home/dmeyer/bindingQTL_share/24a-hNIP-control-tdp/allelic/24a-hNIP-control-tdp_ip_allelic.out',
               '/gpfs/commons/home/dmeyer/bindingQTL_share/24a-hNIP-c9-tdp/allelic/24a-hNIP-c9-tdp_ip_allelic.out')

data.frame(
  basename(input_files),
  basename(ip_files)
)

donors <- c('CTRL-NEUHE723FGT-02545-G', 'CASE-NEUFV237VCZ-01369-G', 'CTRL-NEUHE723FGT-02545-G', 'CASE-NEUFV237VCZ-01369-G')
rbps = c("24a-hNIL-control-tdp", "24a-hNIL-c9-tdp", "24a-hNIP-control-tdp", "24a-hNIP-c9-tdp")

i = 2
input_file = input_files[i]
ip_file = ip_files[i]
sample_index <- donors[i]
sum_stats_file <- str_replace(input_file, ".out$", ".sum_stats.tsv")

#vcf_samples <- system(paste("/nfs/sw/bcftools/bcftools-1.9/bin/bcftools query -l", vcf_file), intern=T)
#vcf_cmd <- paste0("/nfs/sw/bcftools/bcftools-1.9/bin/bcftools query -s '", sample_index, "' -f '%CHROM %POS %ID %REF %ALT [%GT]\n' ",vcf_file)
##vcf_cmd <- paste0("/nfs/sw/bcftools/bcftools-1.9/bin/bcftools query -f '%CHROM %POS %ID %REF %ALT [%GT]\n' ",vcf_file)
#vcf <- fread(cmd=vcf_cmd, header = F, sep=" ", col.names = c("chrom", "position", "variantID", "refAllele", "altAllele", "gt"))
#vcf$refCount <- 0
#vcf$altCount <- sapply(vcf$gt, function(x) { switch(x, "0|0"=0, "0|1"=1, "1|0"=1, "1|1"=2, "0/0"=0, "0/1"=1, "1/0"=1, "1/1"=2, 0) })
#vcf$refCount <- 2 - vcf$altCount
#vcf_case = vcf
#vcf_ctrl = vcf
#write_feather(vcf_ctrl, "/gpfs/commons/home/dmeyer/bindingQTL_share/genotype/CTRL-NEUHE723FGT-02545-G.genotype.feather")
#write_feather(vcf_case, "/gpfs/commons/home/dmeyer/bindingQTL_share/genotype/CASE-NEUFV237VCZ-01369-G.genotype.feather")
vcf_case <- read_feather("/gpfs/commons/home/dmeyer/bindingQTL_share/genotype/CTRL-NEUHE723FGT-02545-G.genotype.feather")
vcf_ctrl <- read_feather("/gpfs/commons/home/dmeyer/bindingQTL_share/genotype/CASE-NEUFV237VCZ-01369-G.genotype.feather")

vcf_combined <-
  vcf_case%>%filter(gt %in% c("0/1", "1/0"))%>%
  inner_join(vcf_ctrl%>%filter(gt %in% c("0/1", "1/0")),
             by=c('chrom','position', 'variantID', 'refAllele', 'altAllele')
  )

het_snps <- vcf_combined$variantID

registerDoMC(4)

i = 3

res <- list()
for (i in 1:4) {
if (i == 1 || i == 3) {
  vcf = vcf_ctrl
} else {
  vcf = vcf_case 
}
input_file = input_files[i]
ip_file = ip_files[i]
sample_index <- donors[i]
sum_stats_file <- str_replace(input_file, ".out$", ".sum_stats.tsv")
input = read_tsv(input_file)%>%rename(chrom=1)
res[[i]] <- nrow(input)
input = input[(input$variantID != "."),]
#input = input[(input$variantID %in% vcf[vcf$altCount == 1,]$variantID),]
input = input[(input$variantID %in% het_snps),]
res[[i]] <- c(res[[i]],  nrow(input))

input = input[input$totalCount >= 10,]
res[[i]] <- c(res[[i]],  nrow(input))

ip_data = read_tsv(ip_file)%>%rename(chrom=1)%>%
    filter(totalCount >= 30)
joined=inner_join(input, ip_data, by = join_by(chrom, position, variantID, refAllele, altAllele))
res[[i]] <- c(res[[i]],  nrow(joined))

input <- input[input$variantID %in% joined$variantID,]
sum_stats = foreach(i = 1:nrow(input), .combine = bind_rows) %dopar% {
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

p = 0.5
epsilon = 0.3
#ci = sum_stats$or_mean
idx = apply(sum_stats, 1, function(x) {
        (x['ci1'] >= p - epsilon) &
        (x['ci1'] <= x['ci2']) & 
        (x['ci2'] <= p + epsilon)
})

res[[i]] <- c(res[[i]],  sum(idx))

filtered_input_file <- str_replace(input_file, "out$", paste0("10readsInput_filtered_epsilon",epsilon,"_sharedhet.txt"))
#filtered_input_file <- str_replace(input_file, "out$", paste0("filtered_epsilon",epsilon,".txt"))
write_tsv(input[idx,], filtered_input_file)
}

IPs <- lapply(ip_files, function(x) {
  read_tsv(x)%>%rename(chrom=1)
})

Inputs <- lapply(input_files, function(x) {
  read_tsv(input_file)%>%rename(chrom=1)
})

df <- data.frame(
  value = c(Inputs[[1]]$totalCount,
            Inputs[[2]]$totalCount,
            Inputs[[3]]$totalCount,
            Inputs[[4]]$totalCount),
  group = rep(rbps,
              c(nrow(Inputs[[1]]),
                nrow(Inputs[[2]]),
                nrow(Inputs[[3]]),
                nrow(Inputs[[4]]))))

ggplot(df, aes(x = value, fill = group))+
  geom_histogram(bins = 10, position=position_dodge(width = 0.2), width = 0.1)+
  scale_x_log10()+
  scale_y_log10()+
  labs(color = "Sample", x = "totalReads input")


df <- data.frame(
  value = c(IPs[[1]]$totalCount,
            IPs[[2]]$totalCount,
            IPs[[3]]$totalCount,
            IPs[[4]]$totalCount),
  group = rep(rbps,
              c(nrow(IPs[[1]]),
                nrow(IPs[[2]]),
                nrow(IPs[[3]]),
                nrow(IPs[[4]]))))
ggplot(df, aes(x = value, fill = group))+
  geom_histogram(bins = 10, position=position_dodge(width = 0.2), width = 0.1)+
  scale_x_log10()+
  scale_y_log10()+
  labs(color = "Sample", x = "totalReads IP")

