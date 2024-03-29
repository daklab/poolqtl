suppressPackageStartupMessages(library(VGAM))
suppressPackageStartupMessages(library(arrow))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(qqman))


geno_fix <- function(x) {
    switch(x,
           "0|0" = 0,
           "1|0" = 0.5,
           "0|1" = 0.5,
           "1|1" = 1,
           "0/0" = 0,
           "1/0" = 0.5,
           "0/1" = 0.5,
           "1/1" = 1,
           "0 0" = 0,
           "1 0" = 0.5,
           "0 1" = 0.5,
           "1 1" = 1,
           "00" = 0,
           "10" = 0.5,
           "01" = 0.5,
           "11" = 1,
            NA)
}

get_cell_lines_from_vcf <- function(vcf_file) {
    cell_lines <- system(paste("bcftools query -l", vcf_file), intern = T)
    return(cell_lines)
}

read_geno_vcf <- function(vcf_file) {
    cell_lines <- get_cell_lines_from_vcf(vcf_file)
    geno = fread(cmd = paste("bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%ID\t%AC\t%AN\t%AF[\t%GT]\n'",
                 vcf_file), header = F, 
                 col.names = c("CHROM", "POS", "REF", "ALT", "ID", "AC", "AN", "AF", cell_lines))
    for (x in cell_lines) {
        geno[[x]] <- sapply(geno[[x]], geno_fix)
    }
    setDT(geno)
    return(geno)
}

read_feather_file <- function(feather_file, cell_lines) {
    geno <- read_feather(feather_file, col_select = NULL, as_data_frame = TRUE, mmap = TRUE)
    setDT(geno)
    geno <- mutate(geno, 
                AC = apply(geno[,..cell_lines], 1, function(x) { sum(x[!is.na(x)]) }), .after = "altAllele")%>%
            mutate(AF = AC / length(cell_lines), .after = "AC")%>%
            mutate(MAF = ifelse(AF > 0.5, 1-AF, AF), .after = "AC")
    return(geno)
}

run_model <- function(geno, cell_lines, obs_file, out_file, deconv_out=NULL) {
    
    #obs_file <- "/home/dmeyer/SecondRound_bQTLs/PCR_optimization_results/ac2/1_0H05_02A6Icahn_Microglia-14cyc_CEBP-beta_hs_i64_allelic_out.txt"
    obs_file <- "/home/dmeyer/SecondRound_bQTLs/PCR_optimization_results/ac2/2_0H06_02A6Icahn_Microglia-16cyc_CEBP-beta_hs_i67_allelic_out.txt"
    #obs_file <- microglia_obs_files[1]
    obs <- fread(obs_file, sep = "\t", header = F, skip = 1) 
    obs <- obs[, 1:(ncol(obs) - 1)]
    colnames(obs) <- unlist(strsplit(readLines(obs_file, 1), "\t"))

    if (!all(obs$variantID %in% geno$variantID)) {
        stop("Allele flipping in", obs_file)
    }
    #if (!all(obs$variantID %in% c(geno$varinatID, geno_flipped$variantID))) {
    #    stop("Not all variants observed in CHIP-seq in VCF")
    #}

    total_thresh <- 30
    maf_thresh <- 0.05
    obs_sub <- obs[(totalCount >= total_thresh) & (totalCount != refCount) & (totalCount != altCount)]
    geno_sub <- geno[(MAF >= maf_thresh)]
    variants_keep <- intersect(obs_sub$variantID, geno_sub$variantID)
    obs_sub <- obs_sub[match(variants_keep, variantID)]
    geno_sub <- geno[match(variants_keep, variantID)]
    
    smoothScatter(obs_sub$refCount, obs_sub$altCount)
    print(ggplot(obs_sub, aes(x = totalCount)) + geom_histogram() +
              labs(title = "Microglia 16 cycle"))
    
    hist(obs$totalCount, main = "Microglia 18 cyc",log='y' )
    #title("Microglia_CEBP_16cyc")
    
    X <- as.matrix(geno[match(variants_keep, variantID), ..cell_lines])
    plot(table(geno$MAF), main = paste(nrow(X), "variants retained with MAF >= 0.05"))
    abline(v=0.05, col='red', lty='dashed')
    

    # observed allelic ratios
    y <- with(obs_sub, altCount / totalCount)
    cat(paste0("len(y): ", length(y),"\n"))

    p1 <- ggplot(obs, aes(x = log10(totalCount))) +  theme_linedraw()+
        geom_histogram(bins = 20) +
        scale_y_log10()+
        geom_vline(xintercept = log10(total_thresh), col='red', lty='dashed')+
        labs(title = paste0(nrow(obs[totalCount >= 100]), " variants with > ", total_thresh, " observations"))

    stopifnot(all(!is.na(y)))
    stopifnot(all(is.numeric(y)))

    conv_fit <- lm(y ~ 0 + X)
    plot(coef(conv_fit), main = "Estimated representations per cell line in pool", xlab = "Cell line index", ylab = "Estimated ratio")
    abline(h = 1/length(cell_lines), col = 'red', lty='dashed')
    if (length(cell_lines) > 1) {
      w <- coef(conv_fit)
    } else {
      w <- 1
    }
    #stopifnot(all(paste0('X', cell_lines) == names(w)))
    names(w) <- cell_lines
    if (!is.null(deconv_out))
      write_tsv(data.table(cell_line=cell_lines, estimated_representation=w), deconv_out)


    afs_expected <- geno_sub$AF

    # Multiply each row of X by w and sum to get predicted AFs
    # so  X %*% x
    mu <- X  %*% w

    p2 <- ggplot(data.frame(mu, afs_expected), aes(x = afs_expected, y = mu)) + 
        geom_point(alpha = 0.5) +
        theme_linedraw()+
        geom_smooth()+
        labs(x = "Allelic ratio if uniform distribution", y = "Predicted allelic ratio")

    # Then we also want to plot predicted allelic ratio against ACTUAL allelic ratio
    observed_allelic_ratio <- obs_sub$altCount / obs_sub$totalCount

    #p3 <- ggplot(data.frame(observed_allelic_ratio, predicted_allelic_ratio = mu),
    #    aes(x = predicted_allelic_ratio,
    #        y = observed_allelic_ratio)) +
    #    geom_point(alpha = 0.5) +
    #    geom_smooth(method=lm, formula = y ~ x, se = TRUE) +
    #    theme_linedraw()+
    #    labs(x = "Baseline allelic ratio predicted (rhat_DNA)", 
    #         y = "Allelic ratio observed in CHiP-seq data (r_IP)", 
    #        title = paste0("R² = ", formatC(cor(observed_allelic_ratio, mu)^2, digits = 3)))

    if (length(cell_lines) == 1) {
      fit <- vglm(cbind(altCount, refCount) ~ 1, betabinomial(lmu = identitylink, lrho = identitylink), obs_sub, crit = "coef")
    } else {
      fit <- vglm(cbind(altCount, refCount) ~ identitylink(offset(mu)), betabinomial(lmu = identitylink, lrho = identitylink), obs_sub, crit = "coef")
    }
    # Uses the mu to predict a 2-dimensional vector of ref and alt
    # Automatically conditions on alt_count + ref_count
    # rho = 1 / (1 + alpha + beta)
    # what david wants conc = (1-rho)/rho
    # rho is not actually concentration (called the correlation parameter)

    ## -----------------------------------------------------------------------------
    # rho if 0 is normal distribution
    cat("Coefficients of fit")
    print(coef(fit, matrix = TRUE))

    rho <- coef(fit, matrix = TRUE)[1,2]
    conc <- (1-rho)/rho


    mu <- as.double(mu)

    pvals <- sapply(seq_along(mu), function(i) {
        crit1 <- obs_sub$totalCount[i]*mu[i]
        crit2 <- obs_sub$altCount[i]
        if (is.na(crit1) | is.na(crit2)) {
        return(NA)
        }
        if (crit2 > crit1) {
            pbetabinom(q = obs_sub$altCount[i], size = obs_sub$totalCount[i], prob = mu[i], rho = rho)
            pval =1-pbetabinom(q = obs_sub$altCount[i]-1, size = obs_sub$totalCount[i], prob = mu[i], rho = rho)
        } else {
            pval = pbetabinom(q = obs_sub$altCount[i], size = obs_sub$totalCount[i], prob = mu[i], rho = rho)
        }
        return(pval)
    })
    qvals <- p.adjust(pvals, method='fdr')
    
    dat <- data.frame(observed_allelic_ratio, predicted_allelic_ratio = mu)
    p4 <- ggplot(dat,
        aes(x = predicted_allelic_ratio,
            y = observed_allelic_ratio)) +
        geom_point(alpha = 0.5, data=dat[qvals >= 0.05,]) +
        geom_point(alpha = 0.5, data=dat[qvals < 0.05,], color="#eb4646") +
        geom_smooth(method=lm, formula = y ~ x, se = TRUE, data=dat) +
        theme_linedraw()+
        labs(x = "Baseline allelic ratio predicted (rhat_DNA)", 
             y = "Allelic ratio observed in CHiP-seq data (r_IP)", 
            title = paste0(sum(qvals < .05, na.rm=TRUE), " SNPs with q < 0.05"))

    grid.arrange(arrangeGrob(p1), ncol=1)
    grid.arrange(arrangeGrob(p2), ncol=1)
    grid.arrange(arrangeGrob(p4), ncol=1)

    hist(-log10(pvals))
    hist(-log10(qvals))

    grid.arrange(arrangeGrob(tableGrob(data.table(rho, conc, `# q < .05` = sum(qvals < 0.05, na.rm=TRUE),
                                                  `# q < .01` = sum(qvals < 0.01, na.rm=TRUE))), ncol=1))
    cat(paste0("Concentration = ", conc, "\n"))
    
    results <- mutate(obs_sub,p_value = pvals,
        qval = qvals,
        direction = sapply(seq_along(mu), function(i) {
            crit1 <- obs_sub$totalCount[i]*mu[i]
            crit2 <- obs_sub$altCount[i]
            if (is.na(crit1) | is.na(crit2)) {
            return(NA)
            }
            if (crit2 > crit1) {
                return("Positive")
            } else { 
                return("Negative")
            }
        })) %>%
        arrange(p_value)
    write_tsv(results, out_file)
    results%>%select(SNP=3, CHR=1, BP=2, P=qval)%>%mutate(CHR=parse_number(CHR))%>%filter(!is.na(P))%>%manhattan
}

## -----------------------------------------------------------------------------
## single-use code goes here
## -----------------------------------------------------------------------------

# Hasn't been necessary
#geno_flipped <- mutate(geno, AC = length(cell_lines) - AC, AF = 1-AF, tmp = refAllele, REF=altAllele)%>%mutate(ALT = tmp)%>%select(-tmp)%>%
#    mutate(ID = paste(contig, position, refAllele, altAllele, sep=":"))

# Run for all monocytes
runMonocytes <- function() {
    monocyte_vcf_file <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/bgzipped_vcfs/monocyte_pool.vcf.gz.gz"
    monocyte_feather <- "~/projects/bqtls/SecondRound_bQTLs/VCF_files/feathers/monocyte_pool.feather"
    monocyte_obs_files <- list.files("~/projects/bqtls/SecondRound_bQTLs/Monocytes", "*mono_allelic_out.txt$", full.names = TRUE)
    monocyte_out_files <- paste0("~/projects/bqtls/SecondRound_bQTLs/asb/Monocyte_", 
                                 basename(monocyte_obs_files)%>%str_extract("^[^_]+")%>%
                                    paste0(".model_results.txt"))
    cell_lines <- get_cell_lines_from_vcf(monocyte_vcf_file)
    geno <- read_feather_file(monocyte_feather, cell_lines)
    pdfs <- str_replace(monocyte_out_files, "model_results.txt$", "model_output.pdf")
    for (i in seq_along(monocyte_obs_files)) {
        if (i == 4) next()
        pdf(pdfs[i])
        cat(paste0("\nrun_model(geno, cell_lines, ",monocyte_obs_files[i], ", ", monocyte_out_files[i], ")\n"))
        withCallingHandlers({
            run_model(geno, cell_lines, monocyte_obs_files[i], monocyte_out_files[i])
        }, error=function(e) print(sys.calls()))
        dev.off()
    }
    rm(geno)
    for (i in 1:10) { invisible(gc()) };
}

# Run for all microglia
runMicroglia <- function() {
    microglia_vcf_file <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/bgzipped_vcfs/merged_ancestries_microglia.vcf.gz"
    microglia_feather <- "~/projects/bqtls/SecondRound_bQTLs/VCF_files/feathers/merged_ancestries_microglia.feather"
    microglia_obs_files <- list.files("~/projects/bqtls/SecondRound_bQTLs/Microglia_All_Ancestries", "*allelic_out.txt$", full.names = TRUE)
    microglia_out_files <- paste0("~/projects/bqtls/SecondRound_bQTLs/asb/Microglia_", basename(microglia_obs_files)%>%str_extract("^[^_]+")%>%paste0(".model_results.txt"))
    cell_lines <- get_cell_lines_from_vcf(microglia_vcf_file)
    geno <- read_feather_file(microglia_feather, cell_lines)
    pdfs <- str_replace(microglia_out_files, "model_results.txt$", "model_output.pdf")
    for (i in 1:length(microglia_obs_files)) {
        if (i == 2) next()
        pdf(pdfs[i])
        cat(paste0("\nrun_model(geno, cell_lines, ",microglia_obs_files[i], ", ", microglia_out_files[i], ")\n"))
        tryCatch({
            run_model(geno, cell_lines, microglia_obs_files[i], microglia_out_files[i])
        }, error=function(e) { message(e) }, 
        finally = function(...) { dev.off(); next(); })
        dev.off()
    }
    rm(geno)
    for (i in 1:10) { invisible(gc()) };
}

runMicroglia <- function() {
    microglia_vcf_file <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/bgzipped_vcfs/merged_ancestries_microglia.vcf.gz"
    microglia_feather <- "~/projects/bqtls/SecondRound_bQTLs/VCF_files/feathers/merged_ancestries_microglia.feather"
    microglia_obs_files <- list.files("~/projects/bqtls/SecondRound_bQTLs/Microglia_All_Ancestries", "*allelic_out.txt$", full.names = TRUE)
    microglia_out_files <- paste0("~/projects/bqtls/SecondRound_bQTLs/asb/Microglia_", basename(microglia_obs_files)%>%str_extract("^[^_]+")%>%paste0(".model_results.txt"))
    cell_lines <- get_cell_lines_from_vcf(microglia_vcf_file)
    geno <- read_feather_file(microglia_feather, cell_lines)
    pdfs <- str_replace(microglia_out_files, "model_results.txt$", "model_output.pdf")
    for (i in 1:length(microglia_obs_files)) {
        if (i == 2) next()
        pdf(pdfs[i])
        cat(paste0("\nrun_model(geno, cell_lines, ",microglia_obs_files[i], ", ", microglia_out_files[i], ")\n"))
        tryCatch({
            run_model(geno, cell_lines, microglia_obs_files[i], microglia_out_files[i])
        }, error=function(e) { message(e) }, 
        finally = function(...) { dev.off(); next(); })
        dev.off()
    }
    rm(geno)
    for (i in 1:10) { invisible(gc()) };
}
runMicrogliaEur <- function() {
    microglia_vcf_file <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/bgzipped_vcfs/microglia_pool_EURonly.vcf.gz"
    microglia_feather <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/feathers/microglia_pool_EURonly.feather"
    microglia_obs_files <- list.files("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/Microglia_European_only/", "*allelic_out.txt$", full.names = TRUE)
    tfs <- basename(microglia_obs_files)%>%str_extract("^[^_]+")
    microglia_out_files <- paste0("~/projects/bqtls/SecondRound_bQTLs/asb/Microglia_EUR_", tfs, ".model_results.txt")
    deconv_out_files <- paste0("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/deconvolve_data/Microglia_EUR_", tfs, ".deconvolution.txt")
    cell_lines <- get_cell_lines_from_vcf(microglia_vcf_file)
    geno <- read_feather_file(microglia_feather, cell_lines)
    pdfs <- str_replace(microglia_out_files, "model_results.txt$", "model_output.pdf")
    
    for (i in 1:length(microglia_obs_files)) {
        if (i == 2) next()
        pdf(pdfs[i])
        cat(paste0("\nrun_model(geno, cell_lines, ",microglia_obs_files[i], ", ", microglia_out_files[i], ")\n"))
        run_model(geno, cell_lines, microglia_obs_files[i], microglia_out_files[i], deconv_out = deconv_out_files[i])
        dev.off()
    }
    rm(geno)
    for (i in 1:10) { invisible(gc()) };
}

runSingleDonor <- function() {
    vcf_file <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/bgzipped_vcfs/microglia_SingleDonor.vcf.gz"
    feather <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/feathers/microglia_SingleDonor.feather"
    obs_files <- list.files("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/Microglia_Single_Donor_PU1/", "*allelic_out.txt$", full.names = TRUE)
    tfs <- "PU1"
    out_files <- paste0("~/projects/bqtls/SecondRound_bQTLs/asb/Microglia_Single_Donor_", tfs, ".model_results.txt")
    deconv_out_files <- paste0("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/deconvolve_data/Microglia_Single_Donor_", tfs, ".deconvolution.txt")
    cell_lines <- get_cell_lines_from_vcf(vcf_file)
    geno <- read_feather_file(feather, cell_lines)
    pdfs <- str_replace(out_files, "model_results.txt$", "model_output.pdf")
    
    for (i in 1:length(obs_files)) {
        if (i == 2) next()
        pdf(pdfs[i])
        cat(paste0("\nrun_model(geno, cell_lines, ",obs_files[i], ", ", out_files[i], ")\n"))
        run_model(geno, cell_lines, obs_files[i], out_files[i], deconv_out = deconv_out_files[i])
        dev.off()
    }
    rm(geno)
    for (i in 1:10) { invisible(gc()) };
}
runMicrogliaPCRExperiment <- function() {
    microglia_vcf_file <- "/home/dmeyer/projects/bqtls/SecondRound_bQTLs/VCF_files/bgzipped_vcfs/merged_ancestries_microglia.vcf.gz"
    microglia_feather <- "~/projects/bqtls/SecondRound_bQTLs/VCF_files/feathers/merged_ancestries_microglia.feather"
    microglia_obs_files <- list.files("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/PCR_optimization_results/allelecounter", "*allelic_out.txt$", full.names = TRUE)
    tfs <- basename(microglia_obs_files)%>%str_extract("1[64]cyc_CEBP")
    deconv_out_files <- paste0("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/PCR_optimization_results/deconvolve_data/Microglia_", tfs, ".deconvolution.txt")
    microglia_out_files <- paste0("~/projects/bqtls/SecondRound_bQTLs/PCR_optimization_results/asb/Microglia_", paste0(tfs,".model_results.txt"))
    cell_lines <- get_cell_lines_from_vcf(microglia_vcf_file)
    #geno <- read_feather_file(microglia_feather, cell_lines)
    pdfs <- str_replace(microglia_out_files, "model_results.txt$", "model_output.pdf")
    for (i in 1:length(microglia_obs_files)) {
        gc()
        pdf(pdfs[i])
        cat(paste0("\nrun_model(geno, cell_lines, ",microglia_obs_files[i], ", ", microglia_out_files[i], ")\n"))
        run_model(geno, cell_lines, microglia_obs_files[i], microglia_out_files[i], deconv_out = deconv_out_files[i])
        tryCatch({
            #run_model(geno, cell_lines, microglia_obs_files[i], microglia_out_files[i])
        }, error=function(e) { message(e) }, 
        finally = function(...) { dev.off(); next(); })
        dev.off()
    }
    #rm(geno)
    #for (i in 1:10) { invisible(gc()) };
}
