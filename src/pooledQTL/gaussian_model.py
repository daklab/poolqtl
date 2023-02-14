import pyro
import torch
from .pyro_utils import BetaReparam, BetaBinomialReparam
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from torch.distributions import constraints
import pyro.poutine as poutine
from . import pyro_utils, asb_data, beta_model
import scipy.stats
import scipy.special
import pandas as pd
import numpy as np
import json
    
import matplotlib.pyplot as plt

logistic = lambda g: 1./(1.+np.exp(-g))

def convertr(hyperparam, name, device): 
    return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = device) 

def setup_effect_prior(name, t_df, scale, num_snps, device): 
    
    if type(t_df) == float and np.isinf(t_df):
        #print("Using normal prior for %s" % name)
        ase = pyro.sample(name, 
            dist.Normal(0., scale).expand([num_snps]).to_event(1)
        )
    elif type(t_df) == float and t_df == 0.: # denotes Laplace
        #print("Using Laplace prior for %s" % name)
        ase = pyro.sample(name, 
            dist.Laplace(0., scale).expand([num_snps]).to_event(1)
        )
    else:
        #print("Using Student-t prior for %s" % name)
        t_df = convertr(t_df, "%s_t_df" % name, device = device)
        ase = pyro.sample(name, 
            dist.StudentT(t_df, 0., scale).expand([num_snps]).to_event(1)
        )
    
    return ase

def normal_model_base(
    data,
    ase_scale = 1.,
    ase_t_df = 3., 
    input_count_conc = 10.,
    asb_scale = 1.,
    asb_t_df = 3., 
    IP_count_conc = 10. 
): # nu ~ gamma(2,0.1)

    device = data.device
    
    ase_scale = convertr(ase_scale, "ase_scale", device = device)
    input_count_conc = convertr(input_count_conc, "input_count_conc", device = device)

    asb_scale = convertr(asb_scale, "asb_scale", device = device)
    IP_count_conc = convertr(IP_count_conc, "IP_count_conc", device = device)
    
    ase = setup_effect_prior("ase", ase_t_df, ase_scale, data.num_snps, device = device)
    asb = setup_effect_prior("asb", asb_t_df, asb_scale, data.num_snps, device = device)

    with pyro.plate("data", data.num_snps):
        input_ratio = torch.logit(data.pred_ratio) + ase
        input_alt = pyro.sample( 
            "input_alt", 
            BetaBinomialReparam(
                torch.sigmoid(input_ratio), 
                input_count_conc,
                total_count = data.input_total_count, 
                eps = 1.0e-8),
            obs = data.input_alt_count)
        IP_alt = pyro.sample( 
            "IP_alt", 
            BetaBinomialReparam(
                torch.sigmoid(input_ratio + asb),
                IP_count_conc,
                total_count = data.IP_total_count,
                eps = 1.0e-8),
            obs = data.IP_alt_count)

def rep_model_base(
    data,
    ase_scale = 1.,
    ase_t_df = 3., 
    input_count_conc = 10.,
    asb_scale = 1.,
    asb_t_df = 3., 
    IP_count_conc = 10.
): # nu ~ gamma(2,0.1)
    
    device = data.device
    
    ase_scale = convertr(ase_scale, "ase_scale", device = device)
    input_count_conc = convertr(input_count_conc, "input_count_conc", device = device)

    asb_scale = convertr(asb_scale, "asb_scale", device = device)
    IP_count_conc = convertr(IP_count_conc, "IP_count_conc", device = device)

    ase = setup_effect_prior("ase", ase_t_df, ase_scale, data.num_snps, device = device)
    asb = setup_effect_prior("asb", asb_t_df, asb_scale, data.num_snps, device = device)

    with pyro.plate("data", data.num_measurements):
        input_ratio = torch.logit(data.pred_ratio) + ase[data.snp_indices]
        input_alt = pyro.sample( 
            "input_alt", 
            BetaBinomialReparam(
                torch.sigmoid(input_ratio), 
                input_count_conc,
                total_count = data.input_total_count, 
                eps = 1.0e-8),
            obs = data.input_alt_count)
        IP_alt = pyro.sample( 
            "IP_alt",
            BetaBinomialReparam(
                torch.sigmoid(input_ratio + asb[data.snp_indices]),
                IP_count_conc,
                total_count = data.IP_total_count,
                eps = 1.0e-8), 
            obs = data.IP_alt_count)

def normal_guide(data, studentT = True, learn_t_dof = True):
    
    device = data.device
    
    def conc_helper(name, init = 5.):    
        param = pyro.param(name + "_param", lambda: torch.tensor(init, device = device), constraint=constraints.positive)
        return pyro.sample(name, dist.Delta(param))
    ase_scale = conc_helper("ase_scale", init = 1.)
    input_count_conc = conc_helper("input_count_conc")
    asb_scale = conc_helper("asb_scale", init = 1.)
    IP_count_conc = conc_helper("IP_count_conc")
    
    z1 = pyro.sample(
        "z1", 
        dist.Normal(
            torch.zeros(data.num_snps, device = device), 
            torch.ones(data.num_snps, device = device)).to_event(1),
        infer={'is_auxiliary': True})
    z2 = pyro.sample(
        "z2", 
        dist.Normal(
            torch.zeros(data.num_snps, device = device),
            torch.ones(data.num_snps, device = device)).to_event(1),
        infer={'is_auxiliary': True})
    ase_loc = pyro.param('ase_loc', lambda: torch.zeros(data.num_snps, device = device))
    ase_sd = pyro.param(
        'ase_sd', 
        lambda: torch.ones(data.num_snps, device = device), 
        constraint=constraints.positive)
    ase = pyro.sample(
        "ase",  
        dist.Delta(
            ase_loc + ase_sd * z1,
            log_density = -ase_sd.log()).to_event(1)) # DAK confirmed that the log_density calculation gives correct variances
    asb_loc = pyro.param('asb_loc', lambda: torch.zeros(data.num_snps, device = device))
    asb_sd = pyro.param(
        'asb_sd', 
        lambda: torch.ones(data.num_snps, device = device), 
        constraint=constraints.positive)
    asb_corr = pyro.param(
        'asb_corr', 
        lambda: torch.zeros(data.num_snps, device = device))
    asb = pyro.sample('asb', dist.Delta(
        asb_loc + asb_corr * z1 + asb_sd * z2,
        log_density = -asb_sd.log()).to_event(1))
    
    to_return = {
        "ase_scale": ase_scale, 
        "input_count_conc": input_count_conc, 
        "asb_scale": asb_scale, 
        "IP_count_conc": IP_count_conc, 
        "ase": ase,
        "asb": asb }
    
    if studentT and learn_t_dof: 
        to_return["ase_t_df"] = conc_helper("ase_t_df", init = 3.)
        to_return["asb_t_df"] = conc_helper("asb_t_df", init = 3.)
        
    return to_return

def fit(
    data, 
    iterations = 1000,
    num_samples = 0,
    use_structured_guide = True,
    learn_concs = True,
    learn_t_dof = True, 
    Laplace = False,
    studentT = True
): 
    
    model_base = rep_model_base if ("Replicate" in str(type(data))) else normal_model_base # better syntax for determining class?
    
    two = torch.tensor(2., device = data.device)
    
    assert not (Laplace and studentT), "Can't be Laplace and StudentT!"
    
    t_dof = 3. if studentT else (0. if Laplace else np.inf)
    
    model = lambda data:  model_base(
        data, 
        ase_scale = dist.HalfCauchy(two) if learn_concs else 1., 
        input_count_conc = dist.Gamma(two,two/10.) if learn_concs else 1.,
        asb_scale = dist.HalfCauchy(two) if learn_concs else 1., 
        IP_count_conc = dist.Gamma(two,two/10.) if learn_concs else 1.,
        ase_t_df = dist.Gamma(two,two/10.) if (studentT and learn_t_dof) else t_dof, 
        asb_t_df = dist.Gamma(two,two/10.) if (studentT and learn_t_dof) else t_dof)

    to_optimize = [
        "ase_scale",
        "input_count_conc",
        "asb_scale",
        "IP_count_conc"]
    
    if studentT and learn_t_dof: to_optimize += [ "ase_t_df", "asb_t_df"]

    if use_structured_guide:
        guide = lambda g: normal_guide(g, studentT = studentT, learn_t_dof = learn_t_dof)
    else: 
        guide = AutoGuideList(model)
        guide.add(AutoDiagonalNormal(poutine.block(model, hide = to_optimize)))
        guide.add(AutoDelta(poutine.block(model, expose = to_optimize)))

    losses = pyro_utils.fit(model,guide,data,iterations=iterations)

    if num_samples > 0: 
        stats,samples = pyro_utils.get_posterior_stats(model, guide, data, num_samples = num_samples, dont_return_sites = ['input_alt','IP_alt'])
    else: 
        stats,samples = None, None

    if use_structured_guide: 
        ase_loc = pyro.param("ase_loc").detach().cpu().numpy() # ~= samples["asb"].mean(0).squeeze().numpy()
        ase_sd = pyro.param("ase_sd").detach().cpu().numpy() # ~= samples["asb"].mean(0).squeeze().numpy()
        
        asb_loc = pyro.param("asb_loc").detach().cpu().numpy() # ~= samples["asb"].mean(0).squeeze().numpy()
        asb_sd = (pyro.param("asb_sd")**2 + pyro.param("asb_corr")**2).sqrt().detach().cpu().numpy() # ~= samples["asb"].std(0).squeeze().numpy()
    else: 
        ase_loc = pyro.param('AutoGuideList.0.loc')[:data.num_snps].detach().cpu().numpy() # confirmed by sampling
        ase_sd = pyro.param('AutoGuideList.0.scale')[:data.num_snps].detach().cpu().numpy()
        
        asb_loc = pyro.param('AutoGuideList.0.loc')[data.num_snps:].detach().cpu().numpy()
        asb_sd = pyro.param('AutoGuideList.0.scale')[data.num_snps:].detach().cpu().numpy()
     
    ase_q = scipy.stats.norm().cdf(-np.abs(ase_loc / ase_sd))
    asb_q = scipy.stats.norm().cdf(-np.abs(asb_loc / asb_sd))
        
    results = pd.DataFrame({
        "variantID" : data.snps,
        "ase_loc" : ase_loc, 
        "ase_sd" : ase_sd, 
        "ase_q" : ase_q,
        "asb_loc" : asb_loc, 
        "asb_sd" : asb_sd, 
        "asb_q" : asb_q
    })
    
    snp_indices = data.snp_indices.detach().cpu().numpy()
    pred_ratio = data.pred_ratio.logit().detach().cpu().numpy()
    
    shrunk = pd.DataFrame({
        "snp_indices" : snp_indices, 
        "shrunk_input_logratio" : pred_ratio + ase_loc[snp_indices],
        "shrunk_IP_logratio" : pred_ratio + ase_loc[snp_indices] + asb_loc[snp_indices]
    })
    
    #both["effect_mean"] = asb_loc[data_both.snp_indices]
    #both["effect_std"] = asb_sd[data_both.snp_indices]
    #both["q"] = q[data_both.snp_indices]

    fit_hypers = { k:pyro.param(
        (k + "_param") if use_structured_guide else ("AutoGuideList.1." + k)
    ).item() for k in to_optimize }

    return losses, model, guide, stats, samples, results, shrunk, fit_hypers



def fit_and_save(dat_here, results_file_stub, device="cpu", plot=False, **kwargs): 

    data = asb_data.RelativeASBdata.from_pandas(dat_here, device = device) 

    losses, model, guide, stats, samples, results, shrunk, fit_hypers = fit(data, **kwargs)
    
    print("Learned hyperparameters:",fit_hypers)
    
    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(losses)
    
    dat_here = pd.concat((
        dat_here.reset_index(drop=True), 
        results.drop(columns = ["variantID"]).reset_index(drop=True),
        shrunk.drop(columns = ["snp_indices"]).reset_index(drop=True)),
        axis = 1 )
    
    dat_here.drop(
        columns = ["input_ratio", "IP_ratio"] # can easily be recalculated from counts
    ).to_csv(results_file_stub.with_suffix(".tsv.gz"), index = False, sep = "\t")
    
    with open(results_file_stub.with_suffix(".json"), "w") as outfile:
        outfile.write(json.dumps(fit_hypers, indent=4))
    
    return dat_here, fit_hypers

def fit_replicates_and_save(df, results_file_stub, device="cpu", **kwargs): 

    assert(len(df)==2)
    df_cat = pd.concat( df, axis = 0)
    data = asb_data.ReplicateASBdata.from_pandas(df_cat)

    losses, model, guide, stats, samples, results, shrunk, fit_hypers = fit(data, **kwargs)

    plt.plot(losses)
    plt.show()
    
    merged_reps = df[0].merge(
        df[1],
        "outer", 
        on = ['contig', 'position', 'position_hg19', 'variantID', 'refAllele', 'altAllele'],
        suffixes=["_1","_2"]
    ).fillna(0, downcast='infer').merge(results, on = "variantID")

    merged_reps.drop(
        columns = ["input_ratio_1", "IP_ratio_1", "input_ratio_2", "IP_ratio_2"]
    ).to_csv(results_file_stub.with_suffix(".tsv.gz"), index = False, sep = "\t")
    
    with open(results_file_stub.with_suffix(".json"), "w") as outfile:
        outfile.write(json.dumps(fit_hypers, indent=4))
    
    for rep_idx in range(1,3): 
        logit_pred_ratio = scipy.special.logit(merged_reps["pred_ratio_%i" % rep_idx])
        merged_reps["shrunk_input_logratio_%i" % rep_idx] = logit_pred_ratio + merged_reps.ase_loc
        merged_reps["shrunk_IP_logratio_%i" % rep_idx] = logit_pred_ratio + merged_reps.ase_loc + merged_reps.asb_loc
    
    return merged_reps, fit_hypers