# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import expon
from scipy.optimize import brentq
from scipy.optimize import fsolve
import pandas as pd

"""# Simulation for Curie-Weiss Model"""

# helper functions for density of Un

from scipy.optimize import fsolve

def find_pii(beta,h):
  if (h != 0) or (beta <= 1):
    pii = fsolve(lambda x: x - np.sqrt(beta)*np.tanh(np.sqrt(beta) * x + h), 2)[0]
  else: # low temperature case
    pii = fsolve(lambda x: x - np.sqrt(beta)*np.tanh(np.sqrt(beta) * x + h), 2)[0]
    pii = [-np.absolute(pii), np.absolute(pii)]
  return pii

def num(x,n,beta,h):
  out = np.exp(-0.5 * x**2 + n * np.log(np.cosh(np.sqrt(beta/n)*x + h)))
  return out


def denom(n,beta,h):
  out = integrate.quad(lambda x: num(x,n,beta,h),-np.inf,+np.inf)[0]
  return out

# rejection sampling for U_n

def rej_sampling(P,Q,M,N,n):

  samples=[]
  i = 0
  while i < N:
    # sample from Q
    x = np.random.default_rng().exponential()*(n**(1/4))
    sign = np.random.binomial(1,0.5) * 2 - 1
    x = x * sign
    # accept with probabiliity P/(M*Q)
    u = np.random.uniform(0,1)
    if u < (P(x)/(M * Q(x))):
      samples.append(x)
      i = i + 1
  samples = np.array(samples)

  return samples

def Un_sampler_approx(n,beta,h,N):
  # approximate by distribution of U
  if h == 0 and beta > 1:
    vroot = find_pii(beta,h)[1]
    sc = 1/np.sqrt(1-beta+vroot**2)
    lc = np.sqrt(n) * vroot
    samples = np.random.normal(loc=0.0, scale = sc, size= N)
    signs = np.random.binomial(1,0.5,N) * 2 - 1
    samples = samples + lc * signs
  elif h != 0:
    vroot = find_pii(beta,h)
    sc = 1/np.sqrt(1-beta+vroot**2)
    lc = np.sqrt(n) * vroot
    samples = np.random.normal(loc=lc, scale = sc, size= N)
  return samples

# Conditional i.i.d sampling for Curie-Weiss

def Q(x,beta,h,n):
  out = expon.pdf(np.absolute(n**(-1/4)*x))/2*n**(-1/4)
  return out

def cw_sampler(n,beta,h,N):

    """
    Generate samples from Curie-Weiss model by the conditional i.i.d method.

    Parameters:
    n: size of each C-W model
    beta, h: parameters for C-W
    N: number of samples

    Returns:
    An N * n array. Each row is a random sample from size-n C-W.
    """

    if (h == 0 and beta <= 1.01): # use rejection_sampling
      dn = denom(n,beta,h)
      X = np.arange(-10,10)
      M = max(np.divide(num(X,n,beta,h)/dn, Q(X,beta,h,n)))
      U = rej_sampling(lambda x: num(x,n,beta,h)/dn,lambda x: Q(x,beta,h,n),M,N,n)
    else: # use approximation
      U = Un_sampler_approx(n,beta,h,N)

    W = np.zeros((N, n))
    P = (np.tanh(np.sqrt(beta/n) * U + h) + 1)/2
    Pmat = np.ones((N, n)) * P[:,None]
    unif = np.random.uniform(0,1,(N, n))
    W = (unif < Pmat)
    W = W * 2 - 1

    return W

"""# Graphon and Potential Outcomes"""

def Gfunc(x1,x2):
    # out = 0.1*x1 + 0.1*x2 + 0.3*x1*x2 + 0.25
    out = 0.5
    return out

Gfunc_vectorized = np.vectorize(Gfunc)

def Yfunc(x1,x2):
    # out = x1**2 + x2**3 + x1*((x2+1)**2)
    # out = x1 + x2 + 1
    out = x1**2 + x1*((x2+1)**2)
    # out = x1 + x1*x2 + 1
    # out = np.exp(x1 * x2)
    # out = x2 * np.exp(x1 * x2)
    return out

# add noise

def YfuncDeriv(x1,x2):
    # out = 3*(x2**2) + 2*x1*(x2+1)
    # out = 1
    out = 2*x1*(x2+1)
    # out = x1
    # out = x1 * np.exp(x1 * x2)
    # out = x1 * 2 * x2 * np.exp(x1 * x2**2)
    return out

def graphon(X, rho, sigma, random_matrix = None):

    N_spins = len(X)
    W = (sigma + 1)/2
    if random_matrix is None:
      random_matrix = np.random.randint(0,65535,(N_spins,N_spins),dtype=np.uint16)

    n_row = []
    n_col = []
    m_row = []
    m_col = []

    for i in range(N_spins):

      if i < N_spins - 1:
        y = X[i+1:]
        w = W[i+1:]
        sig = sigma[i+1:]
        a = X[i]
        y = Gfunc(a,y)*65535
        u = random_matrix[i][i+1:]
        e = y>u
        n_row.append(e.sum())
        m_row.append((e * sig).sum())
      else:
        n_row.append(0)
        m_row.append(0)

      y = X[:i]
      w = W[:i]
      sig = sigma[:i]
      a = X[i]
      y = Gfunc(a,y)*65535
      u = random_matrix[:i, i]
      e = y>u
      n_col.append(e.sum())
      m_col.append((e*sig).sum())

      n = np.array(n_row) + np.array(n_col)
      m = np.array(m_row) + np.array(m_col)
    return m,n,random_matrix

def pout(Yfunc,sigma,mi_vec,ni_vec,noise=None):

  avgi_vec = []

  w = (sigma + 1)/2
  N_spins = len(sigma)

  ratio = mi_vec/ni_vec
  ratio = np.divide(mi_vec, ni_vec, out=np.zeros_like(mi_vec, dtype=np.float64), where=ni_vec!=0)

  yi_vec = Yfunc(sigma, ratio)
  yi_tr_vec = Yfunc(1,ratio)
  yi_ct_vec = Yfunc(-1,ratio)
  avgi_vec = np.average(sigma) - sigma/N_spins

  if noise:
    noise_vec = np.random.normal(0,noise,N_spins)
    yi_vec = yi_vec + noise_vec
    yi_tr_vec = yi_tr_vec + noise_vec
    yi_ct_vec = yi_ct_vec + noise_vec

  return yi_vec,yi_tr_vec,yi_ct_vec,avgi_vec

"""Note: In case N_spins = 10000, random matrix will have size 200MB.

# Point Estimation
"""

def grid_search(sigma,lb = 0,ub = 1,step = 0.01):

  beta = np.concatenate((np.arange(lb,0.95,step),np.arange(0.95,0.99,0.0001),np.arange(0.99,1,0.00001)))
  N_spins = len(sigma)
  avgi_vec = np.average(sigma) - sigma/N_spins

  llh = np.zeros(len(beta))
  for i in range(len(beta)):
    llh[i] = np.sum(np.log(sigma * np.tanh(beta[i] * avgi_vec) + 1))
  id = np.argmax(llh)
  beta_hat = beta[id]

  return beta_hat

# Conventional Hajek Estimator

def Hajek_conventional(yi_vec, sigma_gibbs):
  w_gibbs = (sigma_gibbs + 1)/2
  propensity = 1/2 + 1/2 * np.average(sigma_gibbs)
  tauHajek = np.array(yi_vec,dtype = np.float64) * w_gibbs/propensity - \
             np.array(yi_vec,dtype = np.float64) * (1 - w_gibbs)/(1 - propensity)
  tauHajek = np.average(tauHajek)
  return (tauHajek,propensity)

"""# Oracle Confidence Interval"""

# define nonnormal pdf
def nonnormal_pdf(x):
    out = np.exp(-x**4 / 12) * np.sqrt(2)/(3**(1/4)*special.gamma(1/4))
    return out

# define nonnormal cdf
def nonnormal_cdf(x):
    return integrate.quad(nonnormal_pdf, -np.inf, x)[0]

# define quantile function
def nonnormal_qt(p):
    return brentq(lambda x: nonnormal_cdf(x) - p, -10, 10)  # Adjust the range as necessary

def lin_conventional(Yfunc,YfuncDeriv,beta,h,avg_spin,noise = None):

  pi = 0
  L = YfuncDeriv(1,pi) - YfuncDeriv(-1,pi)
  L2 = L**2
  if noise:
    # L2 += 4 * noise**2
    L2 += 4 * noise

  return L, L2

# find width (for one side) of the CI
def width(N_spins, beta, h, eLn, eLn2, alpha,vroot,etol = 0.0000001):

  if beta < 1 - etol:
    sig = np.sqrt(eLn2 * (np.cosh(np.sqrt(beta) * vroot)**(-2)) + \
                  eLn**2 * beta * (np.cosh(np.sqrt(beta)*vroot+h))**(-4)/(1 - beta + vroot**2))
    width = sig /np.sqrt(N_spins) * norm.ppf(1 - alpha/2)
  elif beta >= 1 - etol:
    sig = eLn
    quant = nonnormal_qt(1 - alpha/2)
    width = N_spins**(-1/4) * sig * quant

  return width

"""# Uniform Confidence Interval"""

# Sample from standard Gaussian
def sample_Gaussian(n_repeat = 1000):
  sample = np.random.normal(0,1,n_repeat)
  return sample

# Sample from W_c for each c.

def sample_Wc(N_spins = 500, lb = 0, ub = 1, step = 0.01, repeat = 1000):

  beta_grid = np.concatenate((np.arange(lb,0.95,step),np.arange(0.95,0.99,0.001),np.arange(0.991,1.0001,0.0001)))
  # beta_grid = np.arange(lb,ub+step,step)
    
  c_grid = np.sqrt(N_spins) * (beta_grid - 1)
  sample = np.zeros((c_grid.shape[0],repeat + 1))

  for i in tqdm(range(len(c_grid))):
    c = c_grid[i]
    beta = beta_grid[i]
    c_abs = np.absolute(c)
    denom = integrate.quad(lambda x: np.exp(-c_abs/2*(x**2)-(x**4)/12), 0, 20)[0]
    # rejection sampling
    W_samples=[]
    j = 0
    while j < repeat:
      # sample from Q
      x = np.random.default_rng().exponential()
      # accept with probabiliity P/(M*Q)
      u = np.random.uniform(0,1)
      X = np.arange(0,10)
      M = max(np.divide(np.exp(-c_abs/2*(X**2)-(X**4)/12)/denom, expon.pdf(X)))
      if u < (np.exp(-c_abs/2*(x**2)-(x**4)/12)/denom)/(M * expon.pdf(x)):
        W_samples.append(x)
        j = j + 1
    W_samples = np.array(W_samples)
    signs = np.random.binomial(1,0.5,repeat)
    signs = signs * 2 - 1
    sample[i,0] = beta
    sample[i,1:] = signs * W_samples

  return sample

def beta_hat_quantile(sample_z, sample_wc, N_spins = 500, alpha = 0.05):
  len_cgrid = sample_wc.shape[0]
  table_quantile = np.zeros((len_cgrid,2))
  table_quantile[:,0] = sample_wc[:,0]
  for i in range(len_cgrid):
    sample_wc_i = sample_wc[i,1:]
    beta = sample_wc[i,0]
    t = sample_z + np.sqrt(beta) * N_spins**(1/4) * sample_wc_i
    stat = 1 - t**(-2) + t**2/(3 * N_spins)
    stat = np.maximum(stat,0)
    stat = np.minimum(stat,1)
    p_zero =  np.average(stat == 0)
    table_quantile[i,1] = np.quantile(stat, p_zero + alpha)
  return table_quantile

def beta_confidence_region(beta_hat, table_quantile = None, N_spins = 500, alpha = 0.05):
  if table_quantile is None:
    sample_z = sample_Gaussian()
    sample_wc = sample_Wc(N_spins)
    table_quantile = beta_hat_quantile(sample_z, sample_wc, N_spins, alpha)
  if beta_hat == 0:
    mask = table_quantile[:,1] >= - 2000 # all TRUE
  else:
    # print("beta_hat = ", beta_hat)
    # print("quantile = ", table_quantile[-1,1])
    mask = beta_hat >= table_quantile[:,1]
  if np.sum(mask) > 0:
    beta_valid = table_quantile[mask,0]
  else:
    print("no valid beta")
    beta_valid = np.array([0])
  return beta_valid

def max_width(beta_valid, z_sample, Wc_sample, beta_hat, N_spins = 500,
              alpha_1 = 0.05, alpha_2 = 0.05, avg_spin = 0.1,
              noise = None, eLn = None, eLn2 = None):

  if len(beta_valid) == 0:
    print("No valid beta.")
    return 0

  width_grid = np.zeros(len(beta_valid))

  for i in range(len(beta_valid)):
    beta = beta_valid[i]
    h = 0
    c = np.sqrt(N_spins) * (beta - 1)
    # mask = (Wc_sample[:,0] == beta)
    mask = np.absolute(Wc_sample[:,0] - beta) < 0.000001

    if np.sum(mask) == 1:
      wc_sample = Wc_sample[mask,1:]
      wc_sample = wc_sample.flatten()

      if eLn is None or eLn2 is None:
        eLn, eLn2 = lin_conventional(Yfunc,YfuncDeriv,beta,h,avg_spin,noise = noise)

      tau_sample = N_spins**(-1/4) * eLn * np.sqrt(beta) * wc_sample + N_spins**(-1/2) * np.sqrt(eLn2) * z_sample

      wd = (np.quantile(tau_sample, 1 - alpha_2/2) - np.quantile(tau_sample, alpha_2/2))/2
      width_grid[i] = wd

    else:
      print("beta = ", beta)
      print("Wc_sample matched = ",Wc_sample[mask,0])
      print("First column of Wc_sample is", Wc_sample[:,0])
      width_grid[i] = 0
  return np.max(width_grid)

"""# Conservative Estimation"""

def local_linear_regression(x, y, x0, bandwidth):

    weights = 3/4 * np.maximum(1 - ((x - x0)/bandwidth)**2,0)
    W = np.diag(weights)
    X = np.vstack([np.ones_like(x), x - x0]).T
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    y_pred = X @ theta
    residuals = y - y_pred

    return theta, residuals

def jacknife(N_spins,pi,theta_0, theta_1, X,rho,random_matrix,hajek = True, residuals = None):

  # create new samples
  new_w = np.random.binomial(1,(pi+1)/2,N_spins)
  new_sigma = new_w * 2 - 1

  # graphon
  mi_vec,ni_vec,_ = graphon(X, rho, new_sigma, random_matrix)

  # get tau_a_vec and tau_b_vec
  tau_a_vec = np.zeros(N_spins)
  tau_b_vec = np.zeros(N_spins)

  yi_hat_1 = np.vstack([np.ones_like(mi_vec/ni_vec), mi_vec/ni_vec]).T @ theta_1
  yi_hat_0 = np.vstack([np.ones_like(mi_vec/ni_vec), mi_vec/ni_vec]).T @ theta_0
  tau_hat = np.array(yi_hat_1,dtype = np.float64) * new_w/((pi+1)/2) - \
            np.array(yi_hat_0,dtype = np.float64) * (1 - new_w)/(1 - (pi+1)/2)

  for i in range(N_spins):
      tau_a_vec[i] = np.average(np.delete(tau_hat, i))
      y_score = Gfunc(X[i],X)*65535
      u_score = random_matrix[i]
      graph_row_i = y_score > u_score
      mi_deletei = mi_vec - graph_row_i * new_sigma[i]
      ni_deletei = ni_vec - graph_row_i
      yi_hat_1_b = np.vstack([np.ones_like(mi_deletei/ni_deletei), mi_deletei/ni_deletei]).T @ theta_1
      yi_hat_0_b = np.vstack([np.ones_like(mi_deletei/ni_deletei), mi_deletei/ni_deletei]).T @ theta_0
      tau_hat_b = np.array(yi_hat_1_b,dtype = np.float64) * new_w/((pi+1)/2) - \
            np.array(yi_hat_0_b,dtype = np.float64) * (1 - new_w)/(1 - (pi+1)/2)
      tau_b_vec[i] = np.average(tau_hat_b)

  # return sigma^2
  tau_a_avg = np.average(tau_a_vec)
  tau_b_avg = np.average(tau_b_vec)
  eLn2 = np.sum((tau_a_vec - tau_a_avg + tau_b_vec - tau_b_avg)**2)
  if hajek:
    eLn2 = np.sum((tau_b_vec - tau_b_avg)**2)
  eLn2 = eLn2 * N_spins
  eLn = np.sqrt(eLn2)

  if residuals is not None:
    eLn2 += np.std(residuals)**2
  
  return eLn, eLn2

"""# Monte-Carlo"""

### implementation of MCMC methods for the Curie-Weiss Model

def monte_conserve(table_beta_hat = None, z_sample = None, Wc_sample = None,
                   repeat = 1000, iteration = 500,N_spins = 500,h = 0,beta = 0.8,
                   rho = 1, noise = 0.05, alpha_1 = 0.05, alpha_2 = 0.05,
                   bandwidth = None):

  if (z_sample is None) or (Wc_sample is None) or (table_beta_hat is None):
    z_sample = sample_Gaussian(repeat)
    Wc_sample = sample_Wc(N_spins, repeat = repeat)
    table_beta_hat = beta_hat_quantile(sample_z, sample_wc, N_spins, alpha_1)

  mean_vec = np.zeros(iteration)
  tau_vec = np.zeros(iteration)
  tauHajek_vec = np.zeros(iteration)
  beta_hat_vec = np.zeros(iteration)
  widthClassic_vec = np.zeros(iteration)
  widthEst_vec = np.zeros(iteration)
  widthDrift_vec = np.zeros(iteration)
  widthOracle_vec = np.zeros(iteration)
  widthOnestep_vec = np.zeros(iteration)
  widthConserv_vec = np.zeros(iteration)
  widthClassic_jack_vec = np.zeros(iteration)
  widthEst_jack_vec = np.zeros(iteration)
  widthOracle_jack_vec = np.zeros(iteration)
  widthOnestep_jack_vec = np.zeros(iteration)
  theta_0_vec = np.zeros((iteration,2))
  theta_1_vec = np.zeros((iteration,2))
  beta_coverage_vec = np.zeros(iteration)

  # sample c-w
  Sigma = cw_sampler(N_spins,beta,h,iteration)
  W = (Sigma + 1)/2
  mean_vec = np.mean(W, axis=1)

  # width-classic does not depend on estimates

  for i in tqdm(range(iteration)):

    # set random seed
    np.random.seed(i)

    # MCMC
    sigma = Sigma[i,]
    w = W[i,]
    X = np.random.uniform(0,1,N_spins)

    # graphon
    mi_vec, ni_vec, random_matrix = graphon(X, rho, sigma)

    yi_vec,yi_tr_vec,yi_ct_vec,avgi_vec = pout(Yfunc,sigma,mi_vec,ni_vec,noise = noise)
    tau = np.average(np.array(yi_tr_vec) - np.array(yi_ct_vec))
    tau_vec[i] = tau

    # Hajek estimators
    tauHajek,propensity = Hajek_conventional(yi_vec, sigma)
    tauHajek_vec[i] = tauHajek

    # estimate beta
    beta_hat = grid_search(sigma)
    beta_hat_vec[i] = beta_hat
    # print("beta_hat = ", beta_hat)

    # oracle Ln, Ln2
    eLn, eLn2 = lin_conventional(Yfunc,YfuncDeriv,beta,h,np.average(sigma),noise)

    # CI when misspecified to beta = 0
    beta_0 = 0
    h_0 = 0
    wd1 = width(N_spins,beta_0,h_0,eLn,eLn2,alpha_1 + alpha_2,0)
    widthClassic_vec[i] = wd1

    # CI with plug-in beta_hat
    wd2 = width(N_spins,beta_hat,0,eLn,eLn2,alpha_1 + alpha_2,0)
    widthEst_vec[i] = wd2

    # CI with conservative estimation
    beta_valid = beta_confidence_region(beta_hat, table_beta_hat, N_spins, alpha_1)
    coverage = beta in beta_valid
    beta_coverage_vec[i] = coverage
    
    wid = max_width(beta_valid, z_sample, Wc_sample, beta_hat, N_spins,
                    alpha_1, alpha_2, np.average(sigma),
                    noise, eLn, eLn2)
    widthDrift_vec[i] = wid

    # CI with oracle beta
    wid = max_width(np.array([beta]), z_sample, Wc_sample, beta_hat, N_spins,
                    alpha_1, alpha_1 + alpha_2, np.average(sigma),
                    noise, eLn, eLn2)
    widthOracle_vec[i] = wid

    # CI without first step pretesting
    beta_range = np.concatenate((np.arange(0,0.95,0.01),np.arange(0.95,0.995,0.001),np.arange(0.995,1.0001,0.0001)))
    wid = max_width(beta_range, z_sample, Wc_sample, beta_hat, N_spins,
                    alpha_1, alpha_1 + alpha_2, np.average(sigma),
                    noise, eLn, eLn2)
    widthOnestep_vec[i] = wid

    # CI with estimated eLn, eLn2
    bdw_0 = bdw_1 = bandwidth
    if bandwidth is None:
      bdw_0 = np.quantile(np.abs(mi_vec[w == 0])/ni_vec[w == 0],0.3)
      bdw_1 = np.quantile(np.abs(mi_vec[w == 1])/ni_vec[w == 1],0.3)
      # print("pilot bandwidth = ", bdw)
    try:
      theta_0, residuals_0 = local_linear_regression(mi_vec[w == 0]/ni_vec[w == 0], yi_vec[w == 0], 0, bdw_0)
      theta_1, residuals_1 = local_linear_regression(mi_vec[w == 1]/ni_vec[w == 1], yi_vec[w == 1], 0, bdw_1)
    except:
      theta_0, residuals_0 = np.array([0,0]),0
      theta_1, residuals_1 = np.array([0,0]),0
      print("Singular value in local_linear_regression")
    theta_0_vec[i,:] = theta_0
    theta_1_vec[i,:] = theta_1

    # get jacknife estimator
    residuals = np.zeros(N_spins)
    residuals[w == 0] = residuals_0
    residuals[w == 1] = residuals_1
    noise_hat = np.std(residuals)
    eLn_hat, eLn2_hat = jacknife(N_spins,0,theta_0, theta_1, X,rho,random_matrix,hajek = True, residuals=residuals)

    # misspecify beta = 0, estimate eLn, eLn2
    beta_0 = 0
    h_0 = 0
    wid = width(N_spins,beta_0,h_0,eLn_hat,eLn2_hat,alpha_1 + alpha_2,0)
    widthClassic_jack_vec[i] = wid

    # plug-in beta, estimate eLn, eLn2
    wid = width(N_spins,beta_hat,0,eLn_hat,eLn2_hat,alpha_1 + alpha_2,0)
    widthEst_jack_vec[i] = wid

    # drift beta, estimate eLn, eLn2
    wid = max_width(beta_valid, z_sample, Wc_sample, beta_hat, N_spins,
                    alpha_1, alpha_2, np.average(sigma),
                    noise_hat, eLn_hat, eLn2_hat)
    widthConserv_vec[i] = wid

    # one step, estimated eLn, eLn2
    wid = max_width(beta_range, z_sample, Wc_sample, beta_hat, N_spins,
                    alpha_1, alpha_1 + alpha_2, np.average(sigma),
                    noise_hat, eLn_hat, eLn2_hat)
    widthOnestep_jack_vec[i] = wid

    # oracle beta, estimated eLn, eLn2
    wid = max_width(np.array([beta]), z_sample, Wc_sample, beta_hat, N_spins,
                    alpha_1, alpha_1 + alpha_2, np.average(sigma),
                    noise_hat, eLn_hat, eLn2_hat)
    widthOracle_jack_vec[i] = wid    

  # Combine all vectors into a DataFrame
  data = {
      'mean_vec': mean_vec,
      'tau_vec': tau_vec,
      'tauHajek_vec': tauHajek_vec,
      'beta_hat_vec': beta_hat_vec,
      'widthClassic_vec': widthClassic_vec,
      'widthEst_vec': widthEst_vec,
      'theta_0_vec_first': theta_0_vec[:,0],
      'theta_0_vec_second': theta_0_vec[:,1],
      'theta_1_vec_first': theta_1_vec[:,0],
      'theta_1_vec_second': theta_1_vec[:,1],
      'widthDrift_vec': widthDrift_vec,
      'widthOracle_vec':widthOracle_vec,
      'widthOnestep_vec':widthOnestep_vec,
      'widthConservative_vec': widthConserv_vec,
      'widthOracle_jack_vec':widthOracle_jack_vec,
      'widthOnestep_jack_vec':widthOnestep_jack_vec,
      'widthClassic_jack_vec': widthClassic_jack_vec,
      'widthEst_jack_vec': widthEst_jack_vec,
      'beta_coverage_vec':beta_coverage_vec
  }

  df = pd.DataFrame(data)

  return df

"""# Beta Grid Experiment"""

def main(idx):
  
  print(f"Running job {idx}")

  # N_spins_grid = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000])
  repeat = 1000
  iteration = 5000
  N_spins = 500
  # N_spins = N_spins_grid[idx]
  alpha_1 = 0.025
  alpha_2 = 0.075
  z_sample = sample_Gaussian(repeat)
  Wc_sample = sample_Wc(N_spins, repeat = repeat)
  table_beta_hat = beta_hat_quantile(z_sample, Wc_sample, N_spins, alpha_1)

  # beta_grid_1 = np.arange(0,1.1,0.1)
  # delta_grid = np.arange(2,5.2,0.2)
  # beta_grid_2 = 1 - 0.1** delta_grid
  # beta_grid = np.concatenate((beta_grid_1,beta_grid_2))
  beta_grid_1 = np.arange(0,1.1,0.1)
  beta_grid_2 = np.array([0.95,0.99,0.995,0.999,0.9999])
  beta_grid = np.concatenate((beta_grid_1, beta_grid_2))

  beta = beta_grid[idx]
  # beta = 0

  np.random.seed(0)

  beta_str = f"{int(beta*1000000):01d}"

  df = monte_conserve(table_beta_hat, z_sample, Wc_sample,
                   repeat = repeat, iteration = iteration,N_spins = N_spins,h = 0,beta = beta,
                   rho = 0.5, noise = 0.05, alpha_1 = alpha_1, alpha_2 = alpha_2,
                   bandwidth = None)
  df.to_csv('simulation_outputs/beta_{0}_size_{1}.csv'.format(beta_str,N_spins), index=False)

if __name__ == "__main__":
    
    from multiprocessing import Pool
    
    with Pool(8) as p:
      p.map(main, range(8,16))
