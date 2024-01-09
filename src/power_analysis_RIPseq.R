
require(tidyverse)
require(doMC)

ase_dir = "/gpfs/commons/home/mschertzer/pilot_pool/allelic/"

input_dat = read_tsv(paste0(ase_dir, "input-rep1_allelic.out")) 

ip_dat = read_tsv(paste0(ase_dir, "hnrnpa1-rep1_allelic.out"))

min_ip_total_count = 20 
min_input_total_count = 10 
min_input_per_allele = 5

ip = ip_dat %>% filter(totalCount >= min_ip_total_count) %>% 
    select(-lowMAPQDepth, -lowBaseQDepth, -rawDepth, -otherBases, -position) 
input = input_dat %>% filter(totalCount >= min_input_total_count) %>% 
    select(-lowMAPQDepth, -lowBaseQDepth, -rawDepth, -otherBases, -position) 

joined = inner_join(input, ip, by = c("contig", "variantID", "refAllele", "altAllele"), suffix = c(".input", ".ip") )
joined_sub = joined %>% filter(altCount.input >= min_input_per_allele, refCount.input >= min_input_per_allele)

sum_stats = foreach(i = 1:nrow(joined_sub), .combine = bind_rows) %dopar% {
    here = joined_sub[i,]
    mat = matrix(c(here$refCount.ip, here$altCount.ip, here$refCount.input, here$altCount.input), 2)
    ft = fisher.test(mat)
    lor_ci = log(ft$conf.int)
    tibble( lor_mean = mean(lor_ci),     
            lor_se = (lor_ci[2] - lor_ci[1])/4 )
}

require(ashr)
ash_result = ashr::ash(sum_stats$lor_mean, sum_stats$lor_se)


sample_g = function(size, g) {
    comp = sample.int( n=length(g$pi), size = size, prob = g$pi, replace = T )
    a = g$a[comp]
    a + runif(size) * (g$b[comp]-a)
}

my_rpois = function(x) rpois(length(x), x)



ip_depth = 65109110
input_depth = 51557670



#Hnrnpk rep1	154990093
#Hnrnpk rep2	53430427
#Hnrnpa1	65109110
#IgG rep1	22574176
#IgG rep2	28664074
#input rep1	51557670
#input rep2	52788744
#Rbfox2	50304061

calc_mixsd.unimix = function(m) { sqrt(sum(m$pi * (m$b - m$a)^2) / 12.) }
calc_mixsd.normalmix = function(m) { sqrt(sum(m$pi * m$sd^2)) }

res_ = foreach(true_or = c(1.1), .combine = bind_rows) %do% {
    foreach(ip_scale_factor = c(1,2,5,10,20), .combine = bind_rows) %do% {
    foreach(input_scale_factor = c(1,2,5,10,20), .combine = bind_rows) %do% {

        sim_dat = joined_sub %>% head(5000) %>%
            mutate( input_alt_prop = altCount.input / totalCount.input, 
                                         input_or = altCount.input / refCount.input, 
                               #ip_alt_prop = altCount.ip / totalCount.ip, 
                               totalCount.ip = my_rpois( totalCount.ip * ip_scale_factor ), 
                               totalCount.input = my_rpois( totalCount.input * input_scale_factor ),
                               altCount.input = rbinom(n(), totalCount.input, input_alt_prop),
                               refCount.input = totalCount.input - altCount.input,
                               #input_or = altCount.input / totalCount.input, 
                               #true_or = exp(sample_g(n(), ash_result$fitted_g)), 
                               true_or = true_or, 
                               ip_or = input_or * true_or, 
                               altCount.ip = rbinom(n(), totalCount.ip, ip_or / (ip_or + 1)),
                               refCount.ip = totalCount.ip - altCount.ip )
    
        pv = foreach(i = 1:nrow(sim_dat), .combine = c) %dopar% {
            here = sim_dat[i,]
            mat = matrix(c(here$refCount.ip, here$altCount.ip, here$refCount.input, here$altCount.input), 2)
            fisher.test(mat)$p.value
        }
        q = p.adjust(pv, method="BH")
        #sig = abs(log(sim_dat$true_or)) >= 0.1
        sig = abs(log(sim_dat$true_or)) > 0
        tibble( ip_scale_factor = ip_scale_factor,
                input_scale_factor = input_scale_factor,
                true_or = true_or,
                n = sum(sig),
                fdrp1 = mean(q[sig] < 0.1), 
                nominal_p5 = mean(pv[sig] < 0.05),
                nominal_p05 = mean(pv[sig] < 0.005) )
    }
    }
}

theme_set(theme_bw(base_size = 14))
res %>% filter(true_or == 2) %>% mutate(ip_scale_factor = as_factor(65 * ip_scale_factor),
               input_scale_factor = as_factor(51 * input_scale_factor)) %>% 
    ggplot(aes( ip_scale_factor, fdrp1, fill = input_scale_factor)) + geom_bar(stat="identity", position = "dodge") + ylab("Power") + ylim(0,1) + xlab("# reads for IP (millions)") + guides(fill=guide_legend(title="")) #+ theme(legend.position = "top") # # reads for input (millions)
ggsave("power_analysis.pdf", width=4, height=3)

res_ %>% filter( ip_scale_factor == 10, input_scale_factor == 10)

res %>% filter( ip_scale_factor == 10, input_scale_factor == 10)



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
