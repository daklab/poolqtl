require(tidyverse)
require(doMC)
require(ashr)
#require(data.table)

results <- read_tsv("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/asb/Monocyte_PU1.model_results.txt")
deconv <-     read_tsv("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/deconvolve_data/Monocytes_PU1.deconvolution.txt")
deconv_res <- read_tsv("/home/dmeyer/projects/bqtls/SecondRound_bQTLs/deconvolve_data/Monocytes_PU1.combined_data.txt")

joined <-
  rename(deconv_res, position = 1, contig = 2)%>%
  left_join(
    results,
    #by = c("position", "contig", "variantID")
  )%>%
  select(-lowMAPQDepth, -lowBaseQDepth, -rawDepth, -otherBases, -position,
         -contig_y, -position_y)%>%
  mutate(alt_geno = round(totalCount * pred), # Note pred=predicted baseline alt ratio
         ref_geno = totalCount - alt_geno)
joined_sub <- filter(joined, totalCount >30, maf >= 0.05)
  
sum_stats = foreach(i = 1:nrow(joined_sub), .combine = bind_rows) %dopar% {
    here = joined_sub[i,]
    
    bt = binom.test(c(here$altCount, here$refCount), p = here$pred)
    lor_ci = log(bt$conf.int)
    tibble( lor_mean = mean(lor_ci),     
            lor_se = (lor_ci[2] - lor_ci[1])/4 )
}

ash_result = ashr::ash(sum_stats$lor_mean, sum_stats$lor_se)

sample_g = function(size, g) {
    comp = sample.int( n=length(g$pi), size = size, prob = g$pi, replace = T )
    # Which component we're sampling each beta from
    # beta is a length(size) vector
    a = g$a[comp] # length k vector of the starts; get the correct a for this
    
    # a is lower bound of confidence interval; b is upper bound
    a + runif(size) * (g$b[comp]-a) # sample from uniform starting from a, end at b
}

my_rpois = function(x) rpois(length(x), x)


calc_mixsd.unimix = function(m) { sqrt(sum(m$pi * (m$b - m$a)^2) / 12.) }
calc_mixsd.normalmix = function(m) { sqrt(sum(m$pi * m$sd^2)) }
mylogit <- function(t) {1 / (1+exp(-t))}
invlogit <- function(t) {1 - mylogit(-t)}

res4 = foreach(true_or = c(1.1), .combine = bind_rows) %do% {
    foreach(ip_scale_factor = c(1,2,5,10,20,50), .combine = bind_rows) %do% {
        #
        # TODO:
        #
        #########################################################
        # At what level do we model donor mixture?              #
        #########################################################
        # To simulate the mixture of donors, we should suppose that there is an RV that represents:
        # * Total # of reads observed per donor (estimate by fitting a poisson to the data?)
        # * Mixture proportions of donors (fit a dirichlet?)
        # * trying to model # of cells is maybe too complicated but probably the right level to model at
        #########################################################
        #
        #
        # We skip the deconvolution part for now and directly just
        # provide the altRatio observed in the CHIP-seq data
        # so we're just playing around with # of observed alleles given this
        # actual ratio
        
        sim_dat = joined_sub %>% # @David why is this 5k? A: to run in reasonable time (you should do the full dataset)
            mutate(
                ip_alt_prop = altCount / totalCount, 
                totalCount = my_rpois( totalCount * ip_scale_factor ), 
                
                # @David is the below line meant to be beta?
                # where beta is the true effect size reported by ashr
                # A: pretty much -- in this setup, the effect size is log(OR)
                log_true_or = sample_g(n(), ash_result$fitted_g), 
                
                true_or = exp(true_or), 
                
                # @David is this right (below)? Should it be multiplication?
                # Answer: UH
                #odds_ratio = (altCount / refCount) / (alt_gen / ref_gen),
                #eff_size = log(odds_ratio),
                lor_ip = log_true_or + invlogit(alt_geno / (alt_geno+ref_geno)),
                #alt_p = mylogit(invlogit(ip_alt_prop)+log_true_or),
                alt_p = mylogit(lor_ip),
                altCount = rbinom(n(), totalCount, alt_p),
                refCount = totalCount - altCount,
                
                ip_or = altCount / totalCount,
            )
    
        pv = foreach(i = 1:nrow(sim_dat), .combine = c) %dopar% {
            here = sim_dat[i,]
            
            # This value of pred should be different, right? Perhaps ip_alt_prop + beta
            #bt = binom.test(c(here$altCount, here$refCount), p = here$pred)
            
            #@David not sure what this p= should be
            # Alternatively could be  here$pred
            # Setting to alt_p does not really work
            #res1 bt = binom.test(c(here$altCount, here$refCount), p = mylogit(invlogit(here$true_or)))
            #res2 bt = binom.test(c(here$altCount, here$refCount), p = mylogit(invlogit(here$altCount/here$totalCount + here$true_or)))
            #res3
            bt = binom.test(c(here$altCount, here$refCount), p = here$alt_p)
            bt$p.value
        }
        q = p.adjust(pv, method="BH")
        #sig = abs(log(sim_dat$true_or)) >= 0.1 # Commented from RNA-IP code
        sig = abs(log(sim_dat$true_or)) > 0
        tibble( ip_scale_factor = ip_scale_factor,
                #input_scale_factor = input_scale_factor,
                true_or = true_or,
                n = sum(sig),
                fdrp1 = mean(q[sig] < 0.1), 
                nominal_p5 = mean(pv[sig] < 0.05),
                nominal_p05 = mean(pv[sig] < 0.005) )
    }
}

#res1 generated with true_or
# bt = binom.test(c(here$altCount, here$refCount), p = mylogit(invlogit(here$true_or)))

#res2 generated using simulated altCount, simulated totalCount, and fitted effect size
# bt = binom.test(c(here$altCount, here$refCount), p = mylogit(invlogit(here$altCount/here$totalCount + here$true_or)))

# TODO: plotting see https://www.notion.so/Create-plots-8295f01adb5c4ced8e5edf3c1d53bb0e 
res_ <- res4

theme_set(theme_bw(base_size = 14))
res_ %>% #filter(true_or == 2) %>% 
    mutate(ip_scale_factor = as_factor(65 * ip_scale_factor)) %>% 
    ggplot(aes( ip_scale_factor, fdrp1)) + geom_bar(stat="identity", position = "dodge") + ylab("Power") + ylim(0,1) + xlab("# reads for IP (millions)") + guides(fill=guide_legend(title="")) #+ theme(legend.position = "top") # # reads for input (millions)
ggsave("power_analysis.pdf", width=4, height=3)

res_ %>% filter( ip_scale_factor == 10)

res_ %>% filter( ip_scale_factor == 10)


# TODO: Download peak data from Ariana
peaks = read_tsv("~/../mschertzer/pilot_pool/macs/hnrnpk-consensus.bed", col_names = c("chr","start","end","name","score"))
prop_genome = sum(peaks$end - peaks$start) / 3e9 # 2%
num_snps_in_peaks = floor(3e6 * prop_genome)

mafs = runif(num_snps_in_peaks, .05, .5)

N = 38
u = matrix(runif(num_snps_in_peaks * N), N) 
hap = sweep(u, 2, mafs, FUN = "<")
cs = colSums(hap)


prop_measured = foreach(n=1:19, .combine = c) %dopar% { 
    N = n*2
    u = matrix(runif(num_snps_in_peaks * N), N) 
    hap = sweep(u, 2, mafs, FUN = "<")
    cs = colSums(hap)
    mean( (cs < N) & (cs > 0))
}

tibble(n = 1:19, prop_measured = prop_measured) %>% 
    ggplot(aes(n, 100*prop_measured)) + geom_point() + geom_line() + ylim(0,100) + xlim(0,20) +
    xlab("Number of fibroblast cell lines") + 
    ylab("% common SNPs represented") + ggtitle("Proportion SNPs represented")
ggsave("prop_SNPs.pdf", height=4, width=4)
