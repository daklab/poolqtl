

import scipy.stats
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.distributions import constraints
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro import poutine
import pyro.infer
import pyro.optim
from pyro.ops.indexing import Vindex
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta, init_to_value
from pyro.infer import SVI, Trace_ELBO, config_enumerate, infer_discrete

from . import pyro_utils, asb_data

from scipy.special import logit
        
def log_sigmoid_deriv(x): 
    return F.logsigmoid( x ) + F.logsigmoid( -x )

logistic = lambda g: 1./(1.+np.exp(-g))

@config_enumerate
def ase_model_base(data,
          input_conc = 5.,
          input_count_conc = 10.,
          prior_prob = 0.5):
    """A model just for ASE. In the style of a spike-and-slab prior. Enumeration handles the binary variable of whether there is ASE for a SNP or not. """
    
    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 

    input_conc = convertr(input_conc, "input_conc")
    input_count_conc = convertr(input_count_conc, "input_count_conc")
    
    prior_prob = convertr(prior_prob, "prior_prob") # prior probability of there being ASE
    
    input_ratio = pyro.sample("input_ratio",
        pyro_utils.BetaReparam(data.pred_ratio, input_conc).to_event(1)
    )

    chooser = torch.stack((data.pred_ratio,input_ratio)) # the mean of the betabinomial is picked from this
    N = len(data.alt_count)
    with pyro.plate("data", N):
        assignment = pyro.sample('assignment', dist.Bernoulli(prior_prob)) # whether this SNP has ASE (1) or not (0)
        mu = Vindex(chooser)[assignment.type(torch.long),torch.arange(N, device = data.device)]
        input_alt = pyro.sample( 
            "input_alt", 
            pyro_utils.BetaBinomialReparam(
                mu,
                input_count_conc, # conc param for the likelihood does NOT depend on whether there is ASE or not (since it just corresponds to tech noise)
                total_count = data.total_count), 
            obs = data.alt_count)

@config_enumerate
def asb_model_base(data,
                  input_conc = 5.,
                  input_count_conc = 10.,
                  IP_conc = 5., 
                  IP_count_conc = 10.,
                  prior_prob_ase = 0.1,
                  prior_prob_asb = 0.1):
    """This doesn't run (chooser.shape != IP_ratio.shape), don't understand why."""
    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 

    input_conc = convertr(input_conc, "input_conc")
    input_count_conc = convertr(input_count_conc, "input_count_conc")
    
    prior_prob_ase = convertr(prior_prob_ase, "prior_prob_ase")
    prior_prob_asb = convertr(prior_prob_asb, "prior_prob_asb")

    IP_conc = convertr(IP_conc, "IP_conc")
    IP_count_conc = convertr(IP_count_conc, "IP_count_conc")

    input_ratio = pyro.sample("input_ratio",
        pyro_utils.BetaReparam(data.pred_ratio, input_conc).to_event(1)
    )

    chooser = torch.stack((data.pred_ratio,input_ratio))
    N = data.num_snps
    with pyro.plate("data", N):
        assignment_ase = pyro.sample('assignment_ase', dist.Bernoulli(prior_prob_ase))
        mu = Vindex(chooser)[assignment_ase.type(torch.long),torch.arange(N, device = data.device)]
        input_alt = pyro.sample( 
            "input_alt", 
            pyro_utils.BetaBinomialReparam(
                mu,
                input_count_conc,
                total_count = data.input_total_count), 
            obs = data.input_alt_count)
        
        IP_ratio = pyro.sample("IP_ratio", pyro_utils.BetaReparam(chooser, IP_conc)) # chooser.shape = 2xN, but IP_ratio.shape=N ??? 
        
        assignment_asb = pyro.sample('assignment_asb', dist.Bernoulli(prior_prob_asb))
        #chooser_IP = torch.stack((mu,IP_ratio)) # this doesn't work for some reason
        chooser_IP = torch.stack((data.pred_ratio,input_ratio,IP_ratio)) # Nx4
        mu_IP = Vindex(chooser_IP)[
            torch.where(assignment_asb.type(torch.bool), assignment_ase.type(torch.long)+2, assignment_ase.type(torch.long)),
            torch.arange(N, device = data.device)]
        
        IP_alt = pyro.sample( "IP_alt", 
            pyro_utils.BetaBinomialReparam(
                mu_IP,
                IP_count_conc,
                total_count = data.IP_total_count), 
            obs = data.IP_alt_count)


@config_enumerate
def normal_model_base(data,
          ase_scale = 1.,
          input_count_conc = 10.,
          asb_scale = 1.,
          IP_count_conc = 10.,
          prior_prob_ase = 0.1,
          prior_prob_asb = 0.1,
          structured = False): # nu ~ gamma(2,0.1)
    """Model for simultaneous ASE and ASB estimation, with spike-and-slab type approach
    for both. I think this doesn't currently handle replicates."""
    
    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 

    prior_prob_ase = convertr(prior_prob_ase, "prior_prob_ase")
    prior_prob_asb = convertr(prior_prob_asb, "prior_prob_asb")

    ase_scale = convertr(ase_scale, "ase_scale")
    input_count_conc = convertr(input_count_conc, "input_count_conc")

    asb_scale = convertr(asb_scale, "asb_scale")
    IP_count_conc = convertr(IP_count_conc, "IP_count_conc")

    ase = pyro.sample("ase", # allele specific expression
        dist.Normal(0., ase_scale).expand([data.num_snps]).to_event(1)
    )

    asb = pyro.sample("asb", # allele specific binding (assuming no ase)
        dist.Normal(0., asb_scale).expand([data.num_snps]).to_event(1)
    )
    
    if structured: asb_when_ase = pyro.sample("asb_when_ase", # assuming there is ase
        dist.Normal(0., asb_scale).expand([data.num_snps]).to_event(1) # could have different scale
    )

    chooser_ase = torch.stack((torch.zeros_like(ase),ase))
    #chooser_asb = torch.stack((torch.zeros_like(asb),asb)) # make this structured? 
    
    with pyro.plate("data", data.num_snps):
        
        assignment_ase = pyro.sample('assignment_ase', dist.Bernoulli(prior_prob_ase))
        ase_effect = Vindex(chooser_ase)[assignment_ase.type(torch.long),torch.arange(data.num_snps, device = data.device)]
        
        input_ratio = torch.logit(data.pred_ratio) + ase_effect
        input_alt = pyro.sample( "input_alt", 
                            pyro_utils.BetaBinomialReparam(torch.sigmoid(input_ratio), 
                                                input_count_conc,
                                                total_count = data.input_total_count, 
                                                eps = 1.0e-8), 
                            obs = data.input_alt_count)
        
        assignment_asb = pyro.sample('assignment_asb', dist.Bernoulli(prior_prob_asb))
        if structured: 
            chooser_asb = torch.stack((torch.zeros_like(asb),asb,asb_when_ase))
            #which_asb = Vindex(chooser_asb)[assignment_ase.type(torch.long),torch.arange(data.num_snps, device = data.device)]
            #chooser_asb_2 = torch.stack((torch.zeros_like(which_asb),which_asb))
            assign = torch.where(assignment_asb.type(torch.bool), assignment_ase.type(torch.long)+1, 0)
            mu_asb = Vindex(chooser_asb)[assign,torch.arange(data.num_snps, device = data.device)]
        else: 
            chooser_asb = torch.stack((torch.zeros_like(asb),asb))
            mu_asb = Vindex(chooser_asb)[assignment_asb.type(torch.long),torch.arange(data.num_snps, device = data.device)]
        IP_alt = pyro.sample( "IP_alt", 
                    pyro_utils.BetaBinomialReparam(torch.sigmoid(input_ratio + mu_asb), 
                                        IP_count_conc,
                                        total_count = data.IP_total_count, 
                                        eps = 1.0e-8), 
                    obs = data.IP_alt_count)
        


def ase_guide(data):
    
    device = data.device
        
    def conc_helper(name, constraint=constraints.positive, init = 5.):    
        param = pyro.param(name + "_param", lambda: torch.tensor(init, device = device), constraint=constraint)
        return pyro.sample(name, dist.Delta(param))
    
    prior_prob = conc_helper("prior_prob", constraint = constraints.unit_interval, init = 0.5)
    
    input_conc = conc_helper("input_conc")
    input_count_conc = conc_helper("input_count_conc")
    
    input_ratio_loc = pyro.param(
        'input_ratio_loc', 
        lambda: torch.zeros(data.num_snps, device = device))
    
    input_ratio_scale = pyro.param(
        'input_ratio_scale', 
        lambda: torch.ones(data.num_snps, device = device), 
        constraint=constraints.positive)
    
    input_ratio_logit = pyro.sample(
        "input_ratio_logit", 
        dist.Normal(input_ratio_loc, input_ratio_scale).to_event(1),
        infer={'is_auxiliary': True})
    
    input_ratio = pyro.sample( 
        "input_ratio", 
        dist.Delta(
            torch.sigmoid( input_ratio_logit ), 
            log_density = -log_sigmoid_deriv(input_ratio_logit)).to_event(1) )
    
    return {
        "input_conc": input_conc, 
        "input_count_conc": input_count_conc, 
        "input_ratio": input_ratio,
        "prior_prob" : prior_prob
    }

def asb_guide(data):
    """Guide for Beta ASB model. Not used currently? """
    
    device = data.device
        
    def conc_helper(name, constraint=constraints.positive, init = 5.):    
        param = pyro.param(name + "_param", lambda: torch.tensor(init, device = device), constraint=constraint)
        return pyro.sample(name, dist.Delta(param))
    
    prior_prob_ase = conc_helper("prior_prob_ase", constraint = constraints.unit_interval, init = 0.1)
    prior_prob_asb = conc_helper("prior_prob_asb", constraint = constraints.unit_interval, init = 0.1)
    
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
                                  torch.sigmoid( input_ratio_logit ), 
                                  log_density = -log_sigmoid_deriv(input_ratio_logit)).to_event(1) )
    
    IP_ratio_loc = pyro.param('IP_ratio_loc', lambda: torch.zeros(data.num_snps, device = device))
    IP_ratio_scale = pyro.param('IP_ratio_scale', 
                                   lambda: torch.ones(data.num_snps, device = device), 
                                   constraint=constraints.positive)
    IP_ratio_logit = pyro.sample("IP_ratio_logit", 
                                 dist.Normal(IP_ratio_loc, IP_ratio_scale).to_event(1),
                                infer={'is_auxiliary': True})
    IP_ratio = pyro.sample( "IP_ratio", dist.Delta(
        F.sigmoid( IP_ratio_logit ), 
        log_density = -log_sigmoid_deriv(IP_ratio_logit)
    ).to_event(1))

    return {"input_conc": input_conc, 
            "input_count_conc": input_count_conc, 
            "input_ratio": input_ratio,
            "prior_prob_ase" : prior_prob_ase, 
            "prior_prob_asb" : prior_prob_asb, 
            "IP_conc": IP_conc, 
            "IP_count_conc": IP_count_conc, 
            "IP_ratio": IP_ratio
           }


def normal_asb_guide(data, structured, init_dict = {}):
    """TODO: handle initialization"""
    
    device = data.device
    
    def conc_helper(name, constraint=constraints.positive, init = 5.):    
        param = pyro.param(
            name + "_param", 
            lambda: init_dict.get(name, init).clone(), # need to fix device? #torch.tensor(init_dict.get(name, init), device = device), 
            constraint=constraint)
        return pyro.sample(name, dist.Delta(param))
    
    prior_prob_ase = conc_helper(
        "prior_prob_ase", 
        constraint = constraints.unit_interval, 
        init = 0.1)
    prior_prob_asb = conc_helper(
        "prior_prob_asb", 
        constraint = constraints.unit_interval, 
        init = 0.1)
    
    ase_scale = conc_helper("ase_scale", init = 1.)
    input_count_conc = conc_helper("input_count_conc")
    asb_scale = conc_helper("asb_scale", init = 1.)
    IP_count_conc = conc_helper("IP_count_conc")
    
    z1 = pyro.sample(
        "z1", 
        dist.Normal(
            torch.zeros(data.num_snps, device = device), 
            torch.ones(data.num_snps, device = device)
        ).to_event(1),
        infer={'is_auxiliary': True})
    z2 = pyro.sample(
        "z2", 
        dist.Normal(
            torch.zeros(data.num_snps, device = device),
            torch.ones(data.num_snps, device = device)
        ).to_event(1),
        infer={'is_auxiliary': True})
    ase_loc = pyro.param(
        'ase_loc',
        lambda: init_dict.get("ase", torch.zeros(data.num_snps, device = device)).clone())
    ase_sd = pyro.param(
        'ase_sd', 
        lambda: torch.ones(data.num_snps, device = device), 
        constraint=constraints.positive)
    ase = pyro.sample(
        "ase",  
        dist.Delta(
            ase_loc + ase_sd * z1,
            log_density = -ase_sd.log()
        ).to_event(1))
    
    asb_loc = pyro.param(
        'asb_loc', 
        lambda: init_dict.get('asb_when_ase' if structured else 'asb', torch.zeros(data.num_snps, device = device)).clone())
    asb_sd = pyro.param(
        'asb_sd', 
        lambda: torch.ones(data.num_snps, device = device), 
        constraint=constraints.positive)
    asb_corr = pyro.param(
        'asb_corr', 
        lambda: torch.zeros(data.num_snps, device = device))
    asb = pyro.sample(
        'asb_when_ase' if structured else 'asb', 
        dist.Delta(
            asb_loc + asb_corr * z1 + asb_sd * z2,
            log_density = -asb_sd.log()
        ).to_event(1))
    
    to_return = {"ase_scale": ase_scale, 
            "input_count_conc": input_count_conc, 
            "asb_scale": asb_scale, 
            "IP_count_conc": IP_count_conc, 
            "ase": ase,
            ('asb_when_ase' if structured else 'asb') : asb
           }
    
    if structured: 
        asb_when_not_ase_loc = pyro.param(
            'asb_when_not_ase_loc', 
            lambda: init_dict.get('asb', torch.zeros(data.num_snps, device = device)).clone())  
        asb_when_not_ase_sd = pyro.param(
            'asb_when_not_ase_sd', 
            lambda: torch.ones(data.num_snps, device = device),
            constraint=constraints.positive)
        asb_when_not_ase = pyro.sample(
            "asb", 
            dist.Normal(
                asb_when_not_ase_loc, 
                asb_when_not_ase_sd
            ).to_event(1))
        to_return["asb"] = asb_when_not_ase
    
    return to_return

@config_enumerate
def full_guide(data):
    """Get q(assignments) as a by-product during inference. Not currently used. """
    # Global variables.
    #with poutine.block(hide_types=["param"]):  # Keep our learned values of global parameters.
    guide(data) # need to set this

    # Local variables.
    N = len(data.alt_count)
    with pyro.plate('data', N):
        assignment_probs = pyro.param('assignment_probs', torch.ones(N, device = data.device) / 2,
                                      constraint=constraints.unit_interval)
        pyro.sample('assignment', dist.Bernoulli(assignment_probs))

def test_ase(): 
    pyro.clear_param_store()
    two = torch.tensor(2., device = device)
    learn_concs = True
    learn_prior_prob = True
    model = lambda data:  model_base(
        data, 
        input_conc = dist.Gamma(two,two/10.) if learn_concs else 5., 
        input_count_conc = dist.Gamma(two,two/10.) if learn_concs else 50.,
        prior_prob = dist.Beta(1.,1.) if learn_prior_prob else 0.5,
    )

    guide = AutoDiagonalNormal(poutine.block(model, hide=['assignment']))

    #elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    # elbo.loss(model,my_guide,data) # testing

    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(model, my_guide, adam, loss=pyro.infer.TraceEnum_ELBO(max_plate_nesting=1) ) 
    pyro.clear_param_store()
    losses = []
    for j in range(200):
        loss = svi.step(data)
        losses.append(loss)
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / data.num_snps), end = "\r")
    plt.plot(losses)

    def get_sample(): 
        guide_trace = poutine.trace(my_guide).get_trace(data)  # record the globals
        trained_model = poutine.replay(model, trace=guide_trace)  # replay the globals
        inferred_model = infer_discrete(trained_model, temperature=1, first_available_dim=-2)  # avoid conflict with data plate
        return(poutine.trace(inferred_model).get_trace(data).nodes["assignment"]["value"] )

    samples = torch.stack([ 
        get_sample()
        for _ in range(100) ]) 
    post_mean = samples.mean(0).detach().numpy()
    _ = plt.hist(post_mean,100); plt.show()

    sns.scatterplot(dat.pred_ratio, dat.allelic_ratio, hue = post_mean); plt.show()

    post_stats,_ = pyro_utils.get_posterior_stats(model, guide, data, dont_return_sites = ["assignment"])

    params = pyro.get_param_store().named_parameters()
    params = { k:v for k,v in params }

    #loc = torch.sigmoid(post_stats["input_ratio"]["mean"]).detach().numpy().squeeze()
    loc = torch.sigmoid(params["input_ratio_loc"]).detach().numpy().squeeze()
    plt.scatter(loc, dat.pred_ratio)
    plt.scatter(loc, dat.allelic_ratio)

    plt.scatter(loc - dat.pred_ratio, dat.allelic_ratio - dat.pred_ratio)

    assign_probs = pyro.param('assignment_probs').detach().numpy()
    plt.hist(assign_probs)

def asb_test():
    pyro.clear_param_store()
    
    two = torch.tensor(2., device = device)
    learn_concs = True
    learn_prior_prob = True
    model = lambda data: asb_model_base(
        data, 
        input_conc = dist.Gamma(two,two/10.) if learn_concs else 5., 
        input_count_conc = dist.Gamma(two,two/10.) if learn_concs else 50.,
        prior_prob_ase = dist.Beta(1.,1.) if learn_prior_prob else 0.1,
        prior_prob_asb = dist.Beta(1.,1.) if learn_prior_prob else 0.1,
         IP_conc = dist.Gamma(two,two/10.) if learn_concs else 5., 
         IP_count_conc = dist.Gamma(two,two/10.) if learn_concs else 50.
    )

    guide = AutoDiagonalNormal(poutine.block(model, hide=['assignment','assignment_asb']))

    elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    elbo.loss(model,guide,data) # testing

    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(model, asb_guide, adam, loss=pyro.infer.TraceEnum_ELBO(max_plate_nesting=1) ) 
    pyro.clear_param_store()
    losses = []
    for j in range(200):
        loss = svi.step(data)
        losses.append(loss)
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / data.num_snps), end = "\r")
    plt.plot(losses)


def fit(
    dat,
    learn_likelihood_concs = True,
    learn_concs = True,
    learn_prior_prob = True,
    structured_model = True, # whether to have a different q(asb) when ASE=0. 
    normal_model = True,
    verbose = True, 
    plot = True, 
    iterations = 500, 
    mc_samples = 100,
    lr = 0.03, 
    device = "cpu"
): 

    one = torch.tensor(1., device = device)
    two = torch.tensor(2., device = device)
    
    data = asb_data.RelativeASBdata.from_pandas(dat, device = device)

    if normal_model: 
        model = lambda data: normal_model_base(
            data, 
            input_count_conc = dist.Gamma(two,two/10.) if learn_likelihood_concs else 200.,
            prior_prob_ase = dist.Beta(one,one) if learn_prior_prob else 0.1,
            prior_prob_asb = dist.Beta(one,one) if learn_prior_prob else 0.1,
            IP_count_conc = dist.Gamma(two,two/10.) if learn_likelihood_concs else 200.,
            ase_scale = dist.HalfCauchy(two) if learn_concs else 1., 
            asb_scale = dist.HalfCauchy(two) if learn_concs else 1., 
            structured = structured_model
        )
    else: 
        model = lambda data: asb_model_base( 
            data,
            input_conc = dist.Gamma(two,two/10.) if learn_concs else 1., 
            IP_conc = dist.Gamma(two,two/10.) if learn_concs else 1., 
            input_count_conc = dist.Gamma(two,two/10.) if learn_likelihood_concs else 200.,
            IP_count_conc = dist.Gamma(two,two/10.) if learn_likelihood_concs else 200.,
            prior_prob_ase = dist.Beta(one,one) if learn_prior_prob else 0.1,
            prior_prob_asb = dist.Beta(one,one) if learn_prior_prob else 0.1 
        )

    # figure out sensible initialization
    dat["allelic_ratio_IP"] = (dat.altCount_IP + 0.5) / (dat.totalCount_IP + 1) 
    ase_init = logit(dat.allelic_ratio) - logit(dat.pred_ratio)
    asb_init = logit(dat.allelic_ratio_IP) - logit(dat.pred_ratio)
    asb_when_ase_init = logit(dat.allelic_ratio_IP) - logit(dat.allelic_ratio)

    init_dict = {
        "prior_prob_ase" : 0.1,
        "prior_prob_asb" : 0.1,
        "input_count_conc" : 200., 
        "IP_count_conc" : 200.,
        "ase_scale" : ase_init.std(), 
        "asb_scale" : asb_init.std(), # should we have separate scales for ASB and ASB2? 
        "ase" : ase_init, 
        "asb" : asb_init, 
        "asb_when_ase" : asb_when_ase_init
    }

    init_dict = {k:torch.tensor(v, device = device, dtype = torch.float) for k,v in init_dict.items()}

    # guide = AutoDiagonalNormal(
    #    poutine.block(model, hide=['assignment_ase','assignment_asb']), # we hide the discrete r.v.s
    #    init_loc_fn = init_to_value(values=init_dict)
    #)

    guide = (lambda data: normal_asb_guide(data, structured = structured_model, init_dict = init_dict)
            ) if normal_model else asb_guide

    elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    #elbo.loss(model,guide,data) # testing

    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=elbo ) # use enumeration to sum over discrete LVs
    pyro.clear_param_store()
    losses = []
    for j in range(iterations):
        loss = svi.step(data)
        losses.append(loss)
        if verbose: print("[iteration %04d] loss: %.4f" % (j + 1, loss / data.num_snps), end = "\r")
    if plot: plt.plot(losses); plt.show()

    def get_sample(*nodes):
        """helper function to get discrete samples"""
        guide_trace = poutine.trace(guide).get_trace(data)  # record the globals
        trained_model = poutine.replay(model, trace=guide_trace)  # replay the globals
        inferred_model = infer_discrete(trained_model, temperature=1, first_available_dim=-2)  # avoid conflict with data plate
        trace = poutine.trace(inferred_model).get_trace(data)
        return([trace.nodes[node]["value"] for node in nodes] )

    samples = [get_sample('assignment_ase','assignment_asb') for _ in range(mc_samples)] # sample discrete RVs from approximate posterior

    stacked = [ torch.stack(g) for g in zip(*samples) ]
    kstacked = dict(zip(['assignment_ase','assignment_asb'], stacked))
    post_mean = {k:v.mean(0).detach().cpu().numpy() for k,v in kstacked.items()}
    #post_mean = samples.mean(0).detach().numpy()
    if plot: _ = plt.hist(post_mean['assignment_ase'],100); plt.show()
    if plot: _ = plt.hist(post_mean['assignment_asb'],100); plt.show()

    if plot: sns.scatterplot(dat.pred_ratio, dat.allelic_ratio, hue = post_mean['assignment_ase']); plt.show()

    if plot: sns.scatterplot(dat.allelic_ratio, dat.altCount_IP / dat.totalCount_IP, hue = post_mean['assignment_asb']); plt.show()

    post_stats,_ = pyro_utils.get_posterior_stats(model, guide, data, dont_return_sites = ["assignment_ase", "assignment_asb"])

    if verbose: print("\n".join([ "%s\t\t%.4g" % (k,v["mean"].cpu().item()) for k,v in post_stats.items() if v['mean'].numel()==1 ]))

    return post_stats, post_mean


if False: 
    from pyro.infer.mcmc.api import MCMC
    from pyro.infer.mcmc import NUTS
    pyro.set_rng_seed(2)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=250, warmup_steps=50)
    mcmc.run(data)
    posterior_samples = mcmc.get_samples()
    
