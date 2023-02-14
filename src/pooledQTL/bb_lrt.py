import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pyro_utils import BetaReparam, BetaBinomialReparam
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from torch.distributions import constraints
from . import pyro_utils, asb_data, beta_model
import scipy.stats
import scipy.special
import pandas as pd
import numpy as np
from pyro.infer import SVI, Trace_ELBO
import statsmodels.stats.multitest
    
def f1(x):
    return torch.expm1(x).log()

def f2(x):
    return x + (1 - x.neg().exp()).log()

def inv_softplus(x):
    big = x > torch.tensor(torch.finfo(x.dtype).max).log()
    return torch.where(
        big,
        f2(x.masked_fill(~big, 1.)),
        f1(x.masked_fill(big, 1.)),
    )

class Model(nn.Module):
    
    def __init__(self, init_beta, conc, learn_conc = False, eps = 1e-8): 
        super().__init__()
        self.eps = eps
        uconc = inv_softplus(torch.tensor(conc))
        if learn_conc: 
            self.register_parameter( "uconc", nn.Parameter(uconc) )
        else:
            self.register_buffer( "uconc", uconc )
        self.beta = nn.Parameter(init_beta)
        
    @property
    def conc(self): 
        return self.uconc.softplus()
       
    def forward(self, y): 
        """y is N x replicates x 2, with last dimension being [altcount, refcount]
        Returns the loglikelihood of each observation
        """
        conc = F.softplus(self.uconc)
        bb = BetaBinomialReparam(self.beta.sigmoid()[:,None], conc, total_count = y.sum(2), eps = self.eps)
        return bb.log_prob(y[:,:,0]) # returns the log likelihood

def fit(model, y, iterations = 1500, loss_tol = 1e-5, **adam_kwargs):
    
    if not "lr" in adam_kwargs: adam_kwargs["lr"] = 0.02
    
    optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
    losses = []
    old_loss = np.inf
    for i in range(iterations):
        #print(i,end="\r")
        loss = -model(y).mean() # + 0.01 * model.uconc**2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_item = loss.item()
        print("Iteration %i loss %.3f conc %.2f" % (i, loss_item, F.softplus(model.uconc).item()), end = "\r")
        losses.append(loss_item)
        if np.abs(loss_item - old_loss) < loss_tol: break
        old_loss = loss_item
    return losses

def fitfit(y, conc = 30., learn_conc = False, **kwargs):
    N,P,_ = y.shape
    beta_init = (y[:,:,0].sum(1) / y.sum((1,2))).logit() # smart initialization of mean
    model = Model(beta_init, conc = conc, learn_conc = learn_conc)
    losses = fit(model, y, **kwargs)
    return(model.beta.detach(), model.uconc.detach(), model(y).detach(), losses)

def lrt(df, learn_conc = False, conc = 200., **kwargs):
    """
    df must have columns "refCount_input","altCount_input","refCount_IP","altCount_IP 
    Works for replicates
    """
    
    y = torch.tensor(df.loc[:,["altCount_input","refCount_input","altCount_IP","refCount_IP"]].to_numpy())
    y = y.view(y.shape[0],2,2)
    beta_null,conc_null,loglik_null,losses = fitfit(y, conc = conc, learn_conc = learn_conc, **kwargs)

    y = torch.tensor(df.loc[:,["altCount_input","refCount_input"]].to_numpy())
    y = y.view(y.shape[0],1,2)
    beta_input,conc_input,loglik_full_input,losses = fitfit(y, conc = conc, learn_conc = learn_conc, **kwargs)

    y = torch.tensor(df.loc[:,["altCount_IP","refCount_IP"]].to_numpy())
    y = y.view(y.shape[0],1,2)
    beta_IP,conc_IP,loglik_full_IP,losses = fitfit(y, conc = conc,learn_conc = learn_conc, **kwargs)

    # LRT per SNP
    dev = 2. * (loglik_full_IP.squeeze() + loglik_full_input.squeeze() - loglik_null.sum(1))
    lrtp = scipy.stats.chi2(df = 1).sf(dev)
    
    _,lrtq = statsmodels.stats.multitest.fdrcorrection(lrtp)
    
    return(lrtp, lrtq, beta_input, beta_IP)