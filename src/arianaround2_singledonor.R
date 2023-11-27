source("src/ariana_round2_VGAM.R")


vcf_file <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/bgzipped_vcfs/microglia_SingleDonor.vcf.gz"
feather <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/feathers/microglia_SingleDonor.feather"
obs_files <- list.files("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/Microglia_Single_Donor_PU1/", "*allelic_out.txt$", full.names = TRUE)
tfs <- "PU1"
out_files <- paste0("~/projects/bqtls/SecondRound_bQTLs/asb/Microglia_Single_Donor_", tfs, ".model_results.txt")
deconv_out_files <- paste0("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/deconvolve_data/Microglia_Single_Donor_", tfs, ".deconvolution.txt")
cell_lines <- get_cell_lines_from_vcf(vcf_file)
geno <- read_feather_file(feather, cell_lines)%>%mutate(AC = AC*2)
pdfs <- str_replace(out_files, "model_results.txt$", "model_output.pdf")
    
pdf(pdfs[1])
# geno already defined
# cell_lines already defined
obs_file <- obs_files[1]
out_file <- out_files[1]

obs <- fread(obs_file, sep = "\t", header = F, skip = 1) 
obs <- obs[, 1:(ncol(obs) - 1)]
colnames(obs) <- unlist(strsplit(readLines(obs_file, 1), "\t"))

if (!all(obs$variantID %in% geno$variantID)) {
    stop("Allele flipping in", obs_file)
}
#if (!all(obs$variantID %in% c(geno$varinatID, geno_flipped$variantID))) {
#    stop("Not all variants observed in CHIP-seq in VCF")
#}

# Apparently this happened, so we have to multiply by 2
if (setequal(na.exclude(geno$AC), c(0.0, 0.5, 1.0))) {
  geno$AC <- geno$AC * 2
}
if (!setequal(na.exclude(geno$AC), c(0.0, 1.0, 2.0))) {
  stop("geno AC is out of whack. Should be count of alternate allele")
}

# filter read counts
total_thresh <- 30
obs_sub <- obs[(totalCount >= total_thresh) & (totalCount != refCount) & (totalCount != altCount)]
# MAF threshold doesn't make sense here but we definitely only want HET SNPs
geno_sub <- geno[(!is.na(AC)) & (AC == 1)]
variants_keep <- intersect(obs_sub$variantID, geno_sub$variantID)
obs_sub <- obs_sub[match(variants_keep, variantID)]
geno_sub <- geno[match(variants_keep, variantID)]
X <- as.matrix(geno[match(variants_keep, variantID), ..cell_lines])

plot(table(geno$AC), main = paste(nrow(X), "variants retained with AC >= 0 and totalCount >= 30"))
abline(v=0.5, col='red', lty='dashed')
abline(v=1.5, col='red', lty='dashed')

if ( any(is.na(afs_expected)) ||  any(is.na(obs_sub$altCount)) ||  any(is.na(obs_sub$refCount))) {
  message("You've got NAs")
}
  

afs_expected <- geno_sub$AF
if (!all(afs_expected == 0.5))
  stop("All AFs should be 0.5 for single-donor, as we're filtering to only Het SNPs")
observed_afs <- with(obs_sub, altCount / totalCount)

hist(observed_afs, main = "Histogram of observed AFs in ChIP-seq data\nExpected = 0.5 (red line), Mean = "%>%paste0(round(mean(observed_afs),5)))
abline(v=0.5, col = 'red', lty='dashed', lwd=3)

fit <- vglm(cbind(altCount, refCount) ~ 1, betabinomial(lmu = identitylink, lrho = identitylink), obs_sub, crit = "coef")


#cat("Coefficients of fit")
#print(coef(fit, matrix = TRUE))

rho <- coef(fit, matrix = TRUE)[1,2]
conc <- (1-rho)/rho
#print(paste0("conc = ",conc))

pvals <- sapply(seq(1,nrow(obs_sub)), function(i) {
    crit1 <- obs_sub$totalCount[i]*0.5
    crit2 <- obs_sub$altCount[i]
    if (is.na(crit1) | is.na(crit2)) {
       return(NA)
    }
    p1 = 1-pbetabinom(q = obs_sub$altCount[i]-1, size = obs_sub$totalCount[i], prob = 0.5, rho = rho)
    p2 = pbetabinom(q = obs_sub$altCount[i], size = obs_sub$totalCount[i], prob = 0.5, rho = rho)
    pval = 2* min(p1, p2)
    return (pval)
})


qvals <- p.adjust(pvals, method = "fdr")

#i = 3634
#crit1 <- obs_sub$totalCount[i]*0.5
#crit2 <- obs_sub$altCount[i]
#if (is.na(crit1) | is.na(crit2)) {
#    next()
#}
#if (crit2 > crit1) {
#    pbetabinom(q = obs_sub$altCount[i], size = obs_sub$totalCount[i], prob = 0.5, rho = rho)
#    pval =1-pbetabinom(q = obs_sub$altCount[i]-1, size = obs_sub$totalCount[i], prob = 0.5, rho = rho)
#    .title = paste0("P(x > ", round(crit2,2), ") = ", format(pval, scientific = TRUE, digits = 3))
#} else {
#    pval = pbetabinom(q = obs_sub$altCount[i], size = obs_sub$totalCount[i], prob = 0.5, rho = rho)
#    .title = paste0("P(x <= ", round(crit2,2), ") = ", format(pval, scientific = TRUE, digits = 3))
#}
#
#p=rbetabinom(9000, obs_sub$totalCount[i], 0.5)
#hist(p, main = .title, xlim = c(min(c(crit2, min(p))), max(crit2, max(p))))
##plot(density(p, from = 0, width=3), main = .title, xlim = c(min(c(crit2, min(p))), max(crit2, max(p))))
#abline(v = crit1, col = 'red', lty='dashed')
#abline(v = crit2, col = 'blue', lty='dashed')


hist(pvals)
hist(-log10(pvals))
hist(qvals)
## ------------------------------------------------------------------------------------------------------------------------------------------------
hist(-log10(qvals))
data.frame(i = 1:length(qvals), q = qvals)%>%arrange(q)%>%
  head(10)%>%
  tableGrob%>%
  arrangeGrob(ncol = 1)%>%
  grid.arrange



hist(pvals)
abline(v = 0.05, col='red', lty='dashed')

hist(-log10(pvals))
abline(v = -log10(0.05), col='red', lty='dashed')

grid.arrange(arrangeGrob(tableGrob(data.table(rho, conc, `# q < .05` = sum(qvals < 0.05, na.rm=TRUE),
                                              `# q < .01` = sum(qvals < 0.01, na.rm=TRUE))), ncol=1))


## ------------------------------------------------------------------------------------------------------------------------------------------------
results <- mutate(obs_sub,p_value = pvals,
        qval = qvals,
    direction = sapply(1:nrow(obs_sub), function(i) {
        crit1 <- obs_sub$totalCount[i]*0.5
        crit2 <- obs_sub$altCount[i]
        if (is.na(crit1) | is.na(crit2)) {
           return(NA)
        }
        if (crit2 > crit1) {
            return("Positive")
        } else { 
            return("Negative")
        }
    }))%>%
arrange(p_value)
write_tsv(results, out_file)

results%>%select(SNP=3, CHR=1, BP=2, P=qval)%>%mutate(CHR=parse_number(CHR))%>%filter(!is.na(P))%>%manhattan
dev.off()
