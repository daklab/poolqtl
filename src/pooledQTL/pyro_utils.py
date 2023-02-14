import pyro
from pyro.infer import Predictive
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import torch

def BetaReparam(mu, conc): 
    return dist.Beta(
        concentration1 = mu * conc,
        concentration0 = (1.-mu) * conc)

def BetaBinomialReparam(mu, conc, total_count, eps = 0.):
    return dist.BetaBinomial(
        concentration1 = mu * conc + eps, 
        concentration0 = (1.-mu) * conc + eps, 
        total_count = total_count)


def get_posterior_stats(
    model,
    guide, 
    data, 
    num_samples=100,
    return_sites = (),
    dont_return_sites = ()
):

    """ extract posterior samples (somewhat weirdly this is done with `Predictive`) """
    #guide.requires_grad_(False)
    predictive = Predictive(
        model, 
        guide=guide, 
        num_samples=num_samples,
        return_sites = return_sites) 

    samples = predictive(data)
    
    for site in dont_return_sites: 
        if site in samples: 
            del samples[site]

    posterior_stats = { 
        k : {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        } for k, v in samples.items() }

    return posterior_stats, samples

def fit(model, guide, data, lr = 0.03, iterations = 200):
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO() ) 
    pyro.clear_param_store()
    losses = []
    for j in range(iterations):
        loss = svi.step(data)
        losses.append(loss)
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / data.num_snps), end = "\r")
    return losses
