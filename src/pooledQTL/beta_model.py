import pyro
import torch
from . import pyro_utils, asb_data
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from pyro import poutine
from torch.distributions import constraints
import torch.nn.functional as F

import scipy.stats
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

logistic = lambda g: 1./(1.+np.exp(-g))

def model_base(data,
          input_conc = 5.,
          input_count_conc = 10. ):
    
    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 

    input_conc = convertr(input_conc, "input_conc")
    input_count_conc = convertr(input_count_conc, "input_count_conc")

    input_ratio = pyro.sample("input_ratio",
        pyro_utils.BetaReparam(data.pred_ratio, input_conc).to_event(1)
    )

    with pyro.plate("data", len(data.alt_count)):
        input_alt = pyro.sample( "input_alt", 
                            pyro_utils.BetaBinomialReparam(input_ratio, 
                                                input_count_conc,
                                                total_count = data.total_count), 
                            obs = data.alt_count)

def full_model_base(data,
          input_conc = 5.,
          input_count_conc = 10.,
          IP_conc = 5., 
          IP_count_conc = 10. ):
    
    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 

    input_conc = convertr(input_conc, "input_conc")
    input_count_conc = convertr(input_count_conc, "input_count_conc")

    IP_conc = convertr(IP_conc, "IP_conc")
    IP_count_conc = convertr(IP_count_conc, "IP_count_conc")

    input_ratio = pyro.sample("input_ratio",
        pyro_utils.BetaReparam(data.pred_ratio, input_conc).to_event(1)
    )
    
    IP_ratio = pyro.sample("IP_ratio",
        pyro_utils.BetaReparam(input_ratio, IP_conc).to_event(1)
    )

    with pyro.plate("data", data.num_snps):
        input_alt = pyro.sample( "input_alt", 
                            pyro_utils.BetaBinomialReparam(input_ratio, 
                                                input_count_conc,
                                                total_count = data.input_total_count), 
                            obs = data.input_alt_count)
        IP_alt = pyro.sample( "IP_alt", 
                    pyro_utils.BetaBinomialReparam(IP_ratio, 
                                        IP_count_conc,
                                        total_count = data.IP_total_count), 
                    obs = data.IP_alt_count)


def log_sigmoid_deriv(x): 
    return F.logsigmoid( x ) + F.logsigmoid( -x )

def guide_mean_field(data):
    
    device = data.device
        
    def conc_helper(name):    
        param = pyro.param(name + "_param", lambda: torch.tensor(5., device = device), constraint=constraints.positive)
        return pyro.sample(name, dist.Delta(param))
    input_conc = conc_helper("input_conc")
    input_count_conc = conc_helper("input_count_conc")
    IP_conc = conc_helper("IP_conc")
    IP_count_conc = conc_helper("IP_count_conc")
    
    input_ratio_loc = pyro.param('input_ratio_loc', lambda: torch.zeros(data.num_snps, device = device))
    input_ratio_scale = pyro.param('input_ratio_scale', 
                                   lambda: torch.ones(data.num_snps, device = device), 
                                   constraint=constraints.positive)
    input_ratio_logit = pyro.sample("input_ratio_logit", 
                                    dist.Normal(input_ratio_loc, input_ratio_scale).to_event(1),
                                    infer={'is_auxiliary': True})
    
    input_ratio = pyro.sample( "input_ratio", 
                              dist.Delta(
                                  F.sigmoid( input_ratio_logit ), 
                                  log_density = -log_sigmoid_deriv(input_ratio_logit)).to_event(1) )
    
    IP_ratio_loc = pyro.param('IP_ratio_loc', lambda: torch.zeros(data.num_snps, device = device))
    IP_ratio_scale = pyro.param('IP_ratio_scale', 
                                   lambda: torch.ones(data.num_snps, device = device), 
                                   constraint=constraints.positive)
    IP_ratio_logit = pyro.sample("IP_ratio_logit", 
                                 dist.Normal(IP_ratio_loc, IP_ratio_scale).to_event(1),
                                infer={'is_auxiliary': True})
    IP_ratio = pyro.sample( "IP_ratio", dist.Delta(F.sigmoid( IP_ratio_logit ), 
                                                  log_density = -log_sigmoid_deriv(IP_ratio_logit)).to_event(1))
    
    return {"input_conc": input_conc, 
            "input_count_conc": input_count_conc, 
            "IP_conc": IP_conc, 
            "IP_count_conc": IP_count_conc, 
            "input_ratio": input_ratio,
            "IP_ratio": IP_ratio
           }

def structured_guide(data):
    device = data.device
    
    def conc_helper(name):    
        param = pyro.param(name + "_param", lambda: torch.tensor(5., device = device), 
                           constraint=constraints.positive)
        return pyro.sample(name, dist.Delta(param))
    input_conc = conc_helper("input_conc")
    input_count_conc = conc_helper("input_count_conc")
    IP_conc = conc_helper("IP_conc")
    IP_count_conc = conc_helper("IP_count_conc")
    
    z1 = pyro.sample("z1", 
                    dist.Normal(torch.zeros(data.num_snps, device = device), 
                                torch.ones(data.num_snps, device = device)).to_event(1),
                    infer={'is_auxiliary': True})
    z2 = pyro.sample("z2", 
                    dist.Normal(torch.zeros(data.num_snps, device = device), 
                                torch.ones(data.num_snps, device = device)).to_event(1),
                    infer={'is_auxiliary': True})
    input_ratio_loc = pyro.param('input_ratio_loc', lambda: torch.zeros(data.num_snps, device = device))
    input_ratio_scale = pyro.param('input_ratio_scale', 
                                   lambda: torch.ones(data.num_snps, device = device), 
                                   constraint=constraints.positive)
    input_ratio_logit = pyro.sample("input_ratio_logit", 
                                    dist.Delta(input_ratio_loc + input_ratio_scale * z1,
                                              log_density = -input_ratio_scale.log()).to_event(1),
                                    infer={'is_auxiliary': True})
    input_ratio = pyro.sample( "input_ratio", 
                              dist.Delta(
                                  torch.sigmoid( input_ratio_logit ), 
                                  log_density = -log_sigmoid_deriv(input_ratio_logit)).to_event(1) )
    
    IP_ratio_loc = pyro.param('IP_ratio_loc', lambda: torch.zeros(data.num_snps, device = device))
    IP_ratio_scale = pyro.param('IP_ratio_scale', 
                                   lambda: torch.ones(data.num_snps, device = device), 
                                   constraint=constraints.positive)
    IP_ratio_corr = pyro.param('IP_ratio_corr', 
                                   lambda: torch.zeros(data.num_snps, device = device))
    IP_ratio_logit = pyro.sample('IP_ratio_logit', 
                                 dist.Delta(IP_ratio_loc + IP_ratio_corr * z1 + IP_ratio_scale * z2,
                                              log_density = -IP_ratio_scale.log()).to_event(1),
                                infer={'is_auxiliary': True})
    IP_ratio = pyro.sample( "IP_ratio", dist.Delta(torch.sigmoid( IP_ratio_logit ), 
                                                  log_density = -log_sigmoid_deriv(IP_ratio_logit)).to_event(1))
    
    return {"input_conc": input_conc, 
            "input_count_conc": input_count_conc, 
            "IP_conc": IP_conc, 
            "IP_count_conc": IP_count_conc, 
            "input_ratio": input_ratio,
            "IP_ratio": IP_ratio
           }


def fit(data, 
       learn_concs = True, 
      iterations = 1000,
      num_samples = 0,
      use_structured_guide = True
      ):

    two = torch.tensor(2., device = data.device)
    
    model = lambda data:  full_model_base(data, 
         input_conc = dist.Gamma(two,two/10.) if learn_concs else 1., 
         input_count_conc = dist.Gamma(two,two/10.) if learn_concs else 1.,
         IP_conc = dist.Gamma(two,two/10.) if learn_concs else 1., 
         IP_count_conc = dist.Gamma(two,two/10.) if learn_concs else 1.)

    to_optimize = ["input_conc",
                   "input_count_conc",
                   "IP_conc",
                   "IP_count_conc"]

    if use_structured_guide: 
        guide = structured_guide
    else: 
        guide = AutoGuideList(model)
        guide.add(AutoDiagonalNormal(poutine.block(model, hide = to_optimize)))
        guide.add(AutoDelta(poutine.block(model, expose = to_optimize)))

    losses = pyro_utils.fit(model,guide,data,iterations=iterations)

    if num_samples > 0: 
        stats, samples = pyro_utils.get_posterior_stats(model, guide, data, num_samples = num_samples, dont_return_sites = ['input_alt','IP_alt'])
    else: 
        stats, samples = None, None
    
    fit_hypers = { k:pyro.param(k + "_param").item() for k in to_optimize }

    # ~= samples["input_ratio"].logit().mean(0)
    input_ratio_loc = pyro.param("input_ratio_loc").detach().cpu().numpy() 
    # ~= samples["input_ratio"].logit().std(0)
    ase_sd = pyro.param("input_ratio_scale").detach().cpu().numpy() # same as ase_sd

    # ~= samples["IP_ratio"].logit().mean(0)
    IP_ratio_loc = pyro.param("IP_ratio_loc").detach().cpu().numpy() 
    # ~= samples["IP_ratio"].logit().std(0)
    #IP_ratio_sd = torch.sqrt(pyro.param("IP_ratio_scale")**2 + pyro.param("IP_ratio_corr")**2).detach().cpu().numpy() 
    
    logit_pred_ratio = data.pred_ratio.logit()
    # ~= samples["input_ratio"].logit().mean(0) - logit_pred_ratio
    ase_loc = (pyro.param("input_ratio_loc") - logit_pred_ratio).detach().cpu().numpy() 
    ase_q = scipy.stats.norm().cdf(-np.abs(ase_loc / ase_sd)) # ~= (samples["input_ratio"] > samples["IP_ratio"]).float().mean(0)
    
    asb_loc = (pyro.param("IP_ratio_loc") - pyro.param("input_ratio_loc")).detach().cpu().numpy() 
    asb_sd = (pyro.param("IP_ratio_scale")**2 
              + (pyro.param("input_ratio_scale") 
                 - pyro.param("IP_ratio_corr"))**2 ).sqrt().detach().cpu().numpy() 

    asb_q = scipy.stats.norm().cdf(-np.abs(asb_loc / asb_sd)) # ~= (samples["input_ratio"] > samples["IP_ratio"]).float().mean(0)
    
    results = pd.DataFrame({"shrunk_input_logratio" : input_ratio_loc, 
                            "ase_loc" : ase_loc, 
                            "ase_sd" : ase_sd, 
                            "ase_q" : ase_q,
                            "shrunk_IP_logratio" : IP_ratio_loc, 
                            "asb_loc" : asb_loc, 
                            "asb_sd" : asb_sd, 
                            "asb_q" : asb_q
                           })
    
    return losses, model, guide, stats, samples, results, fit_hypers


def make_plots(dat_here, fdr_threshold):
        
    plt.figure(figsize=(14,10))
    plt.subplot(221)
    plt.scatter(dat_here.input_ratio, dat_here.IP_ratio,alpha=0.1, color="gray")
    dat_ss = dat_here[dat_here.asb_q < fdr_threshold]
    plt.scatter(dat_ss.input_ratio, dat_ss.IP_ratio,alpha=0.03, color = "red")
    plt.xlabel("Input proportion alt"); plt.ylabel("IP proportion alt")
    plt.title('%i (%.1f%%) significant %.0f%% FDR' % ((dat_here.asb_q < fdr_threshold).sum(), 
                                                      100. * (dat_here.asb_q < fdr_threshold).mean(), 
                                                      fdr_threshold*100))

    plt.subplot(222)
    plt.scatter( logistic(dat_here.shrunk_input_logratio), logistic(dat_here.shrunk_IP_logratio),alpha=0.1, color="gray")
    dat_ss = dat_here[dat_here.asb_q < fdr_threshold]
    plt.scatter(logistic(dat_ss.shrunk_input_logratio), logistic(dat_ss.shrunk_IP_logratio),alpha=0.03, color = "red")
    plt.xlabel("Shrunk input proportion alt"); plt.ylabel("Shrunk IP proportion alt")
    plt.title('%i (%.1f%%) significant %.0f%% FDR' % ((dat_here.asb_q < fdr_threshold).sum(), 
                                                      100. * (dat_here.asb_q < fdr_threshold).mean(), 
                                                      fdr_threshold*100))

    plt.subplot(223)
    plt.scatter( dat_here.pred_ratio, dat_here.input_ratio,alpha=0.1, color="gray")
    dat_ss = dat_here[dat_here.ase_q < fdr_threshold]
    plt.scatter( dat_ss.pred_ratio, dat_ss.input_ratio,alpha=0.03, color = "red")
    plt.xlabel("Alt proportion in DNA"); plt.ylabel("Input proportion alt")
    plt.title('%i (%.1f%%) significant %.0f%% FDR' % ((dat_here.ase_q < fdr_threshold).sum(), 
                                                      100. * (dat_here.ase_q < fdr_threshold).mean(), 
                                                      fdr_threshold*100))

    plt.subplot(224)
    plt.scatter( dat_here.pred_ratio, logistic(dat_here.shrunk_input_logratio),alpha=0.1, color="gray")
    dat_ss = dat_here[dat_here.ase_q < fdr_threshold]
    plt.scatter(dat_ss.pred_ratio, logistic(dat_ss.shrunk_input_logratio),alpha=0.03, color = "red")
    plt.xlabel("Alt proportion in DNA"); plt.ylabel("Shrunk input proportion alt")
    plt.title('%i (%.1f%%) significant %.0f%% FDR' % ((dat_here.ase_q < fdr_threshold).sum(), 
                                                      100. * (dat_here.ase_q < fdr_threshold).mean(), 
                                                      fdr_threshold*100))
    plt.show()
    return()


def fit_plot_and_save(dat_here, results_file, fdr_threshold = 0.05, device="cpu", **kwargs): 
    
    data = asb_data.RelativeASBdata.from_pandas(dat_here, device = device)
    
    losses, model, guide, stats, samples, results, fit_hypers = fit(data, **kwargs)

    print("Learned hyperparameters:",fit_hypers)
    
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    
    dat_here = pd.concat((dat_here.reset_index(drop=True), 
                          results.reset_index(drop=True)), axis = 1 )
    
    dat_here.drop(columns = ["input_ratio", "IP_ratio"] # can easily be recalculated from counts
                ).to_csv(results_file, index = False, sep = "\t")

    make_plots(dat_here, fdr_threshold)
    
    return dat_here