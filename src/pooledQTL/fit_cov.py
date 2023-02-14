import pyro.distributions as dist
import torch.distributions as torch_dist 
#from torch.distributions import wishart
#torch.distributions.wishart
from pyro.infer.autoguide import AutoDelta, init_to_value
from pyro.infer import SVI, Trace_ELBO

def model(x,s): 
    N,P = x.shape
    mu = torch.zeros(P, device = x.device)
    
    #theta = pyro.sample("theta", dist.HalfCauchy(torch.ones(P, device = x.device)).to_event(1).mask(False))
    theta = pyro.sample("theta", dist.ImproperUniform(constraints.positive, (), (P,)))
    
    # Lower cholesky factor of a correlation matrix
    L_omega = pyro.sample("L_omega", dist.LKJCholesky(P, torch.tensor(1., device = x.device)))
    
    L_Omega = torch.diag(theta.sqrt()) @ L_omega
    
    cov = L_Omega @ L_Omega.T
    
    with pyro.plate("samples", N): 
        pyro.sample(
            "obs", 
            dist.MultivariateNormal(mu, covariance_matrix = cov + s), 
            obs = x)
        

def fit_cov(
    x, # N x P numpy array
    s, # N x P numpy array of std errors on x
    batchsize = 1000, 
    iterations = 500, 
    loss_tol = 1e-1, 
    adam_opts = {"lr": 0.02}, 
    verbose = True, 
    device = "cpu"):
    """Estimates a covariance and correlation matrix for x accounting for the s.e."""
    
    S = torch.diag_embed(torch.tensor(s, device = device))*2
    X = torch.tensor(x, device = device)
    
    cov = X.T.cov()
    inv_diag_cov_sqrt = 1. / cov.diag().sqrt()
    cor = cov * inv_diag_cov_sqrt * inv_diag_cov_sqrt[:,None]

    init_dict = {
        "theta" : X.std(0)**2, 
        "L_omega" : torch.linalg.cholesky(cor)
    }

    guide = AutoDelta(model, init_loc_fn = init_to_value(values = init_dict))
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam(adam_opts)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    losses = []
    old_loss = np.inf
    
    batch_ids = torch.split( torch.arange(X.shape[0]), batchsize)
    for i in range(iterations):
        batch_losses = []
        for j,batch in enumerate(batch_ids): 
            batch_loss = svi.step(X[batch],S[batch])
            batch_losses.append(batch_loss)
            if verbose: print("Batch",i, j, batch_loss, end="\r")
        loss = np.mean(batch_losses)
        losses.append(loss)
        if verbose: print("Epoch",i, loss)
        if np.abs(loss - old_loss) < loss_tol: break
        old_loss = loss
        
        L_omega = pyro.param("AutoDelta.L_omega")
        theta = pyro.param("AutoDelta.theta")
        L_Omega = torch.diag(theta.sqrt()) @ L_omega
        cov = L_Omega @ L_Omega.T
        
        cor = L_omega @ L_omega.T
        
    return losses, cov, cor
        
#x = merged.loc[:,["ase_loc_1","ase_loc_2"]].to_numpy()
#s = merged.loc[:,["ase_sd_1","ase_sd_2"]].to_numpy()
#device = "cuda:0" if torch.cuda.is_available() else "cpu"

#losses, cov, cor = fit_cov(x, s)

#model(X,S)

if __name__ == "__main__":

    from torch.distributions.multivariate_normal import MultivariateNormal

    N = 1000
    true_cov = torch.Tensor([[2., 1.], [1., 2.]])
    se = np.sqrt(0.1)
    std_error_var = torch.eye(2) * se
    distrib = MultivariateNormal(loc=torch.zeros([N,2]), covariance_matrix=true_cov + std_error_var)
    x = distrib.rsample()
    s = torch.full_like(x, se)
    losses, cov, cor = fit_cov(x, s)
