# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy.stats import chi2
import networkx as nx
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import expon
from numba import njit,prange,objmode
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import brentq
from scipy.optimize import fsolve
from google.colab import drive
import gc
import pandas as pd
import time

from google.colab import drive
drive.mount('/content/drive')

cd 'drive/My Drive'

cd 'interference'

def read_df_from_csv(name):
    file_name = f"{name}.csv"
    # Full path to the file
    folder_name = '/content/drive/My Drive/interference/'
    full_path = folder_name + 'simulation_outputs/' + file_name

    # Read CSV file using numpy
    df = pd.read_csv(full_path)

    return df

"""# Beta Grid Experiment"""

def diagnoistic(df):

  df_new = df.copy()
  mean_vec = df_new['mean_vec']
  tau_vec = df_new['tau_vec']
  tauHajek_vec = df_new['tauHajek_vec']
  beta_hat_vec = df_new['beta_hat_vec']
  widthClassic_vec = df_new['widthClassic_vec']
  widthEst_vec = df_new['widthEst_vec']
  widthDrift_vec = df_new['widthDrift_vec']
  widthOracle_vec = df_new['widthOracle_vec']
  widthOnestep_vec = df_new['widthOnestep_vec']
  widthOracle_jack_vec = df_new['widthOracle_jack_vec']
  widthOnestep_jack_vec = df_new['widthOnestep_jack_vec']
  widthConservative_vec = df_new['widthConservative_vec']
  widthClassic_jack_vec = df_new['widthClassic_jack_vec']
  widthEst_jack_vec = df_new['widthEst_jack_vec']
  tau = np.average(tau_vec) # treatment effect

  # Classical CI
  ec_Classic = np.average((tau < tauHajek_vec + widthClassic_vec) * (tau > tauHajek_vec - widthClassic_vec))
  print("Classic empirical coverage = ", ec_Classic)
  il_Classic = np.average(widthClassic_vec)*2
  print("Classic interval length = ", il_Classic)

  # Plug-in CI
  ec_Plugin = np.average((tau < tauHajek_vec + widthEst_vec) * (tau > tauHajek_vec - widthEst_vec))
  print("Plug-in empirical coverage = ", ec_Plugin)
  il_Plugin = np.average(widthEst_vec)*2
  print("Plugin interval length = ", il_Plugin)

  # Adaptive CI
  ec_Adapt = np.average((tau < tauHajek_vec + widthDrift_vec) * (tau > tauHajek_vec - widthDrift_vec))
  il_Adapt = np.average(widthDrift_vec)*2
  print("Adaptive empirical coverage = ", ec_Adapt)
  print("Adaptive interval length = ", il_Adapt)

  # Oracle CI
  ec_Oracle = np.average((tau < tauHajek_vec + widthOracle_vec) * (tau > tauHajek_vec - widthOracle_vec))
  il_Oracle = np.average(widthOracle_vec)*2
  print("Oracle empirical coverage = ", ec_Oracle)
  print("Oracle interval length = ", il_Oracle)

  # Onestep CI
  ec_Onestep = np.average((tau < tauHajek_vec + widthOnestep_vec) * (tau > tauHajek_vec - widthOnestep_vec))
  il_Onestep = np.average(widthOnestep_vec)*2
  print("Onestep empirical coverage = ", ec_Onestep)
  print("Onestep interval length = ", il_Onestep)

  # Classical CI + jacknife
  ec_Classic_jack = np.average((tau < tauHajek_vec + widthClassic_jack_vec) * (tau > tauHajek_vec - widthClassic_jack_vec))
  print("Classic (with jacknife) empirical coverage = ", ec_Classic_jack)
  il_Classic_jack = np.average(widthClassic_jack_vec)*2
  print("Classic (with jacknife) interval length = ", il_Classic_jack)

  # Plug-in CI + jacknife
  ec_Plugin_jack = np.average((tau < tauHajek_vec + widthEst_jack_vec) * (tau > tauHajek_vec - widthEst_jack_vec))
  print("Plug-in (with jacknife) empirical coverage = ", ec_Plugin_jack)
  il_Plugin_jack = np.average(widthEst_jack_vec)*2
  print("Plugin (with jacknife) interval length = ", il_Plugin_jack)

  # Adaptive CI + jacknife
  ec_Conservative = np.average((tau < tauHajek_vec + widthConservative_vec) * (tau > tauHajek_vec - widthConservative_vec))
  il_Conservative = np.average(widthConservative_vec)*2
  print("Conservative empirical coverage = ", ec_Conservative)
  print("Conservative interval length = ", il_Conservative)

  # Interval Length from Simulation
  il_Simulation = np.quantile(tauHajek_vec,0.95) - np.quantile(tauHajek_vec,0.05)
  print("Simulation interval length = ", il_Simulation)

  # Oracle CI + jacknife
  ec_Oracle_jack = np.average((tau < tauHajek_vec + widthOracle_jack_vec) * (tau > tauHajek_vec - widthOracle_jack_vec))
  il_Oracle_jack = np.average(widthOracle_jack_vec)*2
  print("Oracle (with jacknife) empirical coverage = ", ec_Oracle_jack)
  print("Oracle (with jacknife) interval length = ", il_Oracle_jack)

  # Onestep CI + jacknife
  ec_Onestep_jack = np.average((tau < tauHajek_vec + widthOnestep_jack_vec) * (tau > tauHajek_vec - widthOnestep_jack_vec))
  il_Onestep_jack = np.average(widthOnestep_jack_vec)*2
  print("Onestep (with jacknife) empirical coverage = ", ec_Onestep_jack)
  print("Onestep (with jacknife) interval length = ", il_Onestep_jack)

  return ec_Classic, ec_Plugin, ec_Adapt, ec_Classic_jack, ec_Plugin_jack, ec_Conservative, \
  ec_Oracle, ec_Onestep, ec_Oracle_jack,ec_Onestep_jack, il_Classic, il_Plugin, il_Adapt, \
  il_Classic_jack, il_Plugin_jack, il_Conservative,il_Simulation, il_Oracle, il_Onestep, il_Oracle_jack, il_Onestep_jack

"""# Diagnosis"""

# beta_grid_1 = np.arange(0,1.1,0.1)
# delta_grid = np.arange(2,5.2,0.2)
# beta_grid_2 = 1 - 0.1** delta_grid
# beta_grid = np.concatenate((beta_grid_1,beta_grid_2))
beta_grid_1 = np.arange(0,1,0.1)
beta_grid_2 = np.array([0.95,0.99,0.995,0.999,0.9999,1])
beta_grid = np.concatenate((beta_grid_1, beta_grid_2))

ec_Classic_grid = np.zeros(len(beta_grid))
il_Classic_grid = np.zeros(len(beta_grid))
ec_Plugin_grid = np.zeros(len(beta_grid))
il_Plugin_grid = np.zeros(len(beta_grid))
ec_Adapt_grid = np.zeros(len(beta_grid))
il_Adapt_grid = np.zeros(len(beta_grid))
ec_Conservative_grid = np.zeros(len(beta_grid))
il_Conservative_grid = np.zeros(len(beta_grid))
ec_Classic_jack_grid = np.zeros(len(beta_grid))
il_Classic_jack_grid = np.zeros(len(beta_grid))
ec_Plugin_jack_grid = np.zeros(len(beta_grid))
il_Plugin_jack_grid = np.zeros(len(beta_grid))
il_Simulation_grid = np.zeros(len(beta_grid))
il_Oracle_jack_grid = np.zeros(len(beta_grid))
ec_Oracle_jack_grid = np.zeros(len(beta_grid))
il_Onestep_jack_grid = np.zeros(len(beta_grid))
ec_Onestep_jack_grid = np.zeros(len(beta_grid))

# N_spins_grid = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000])

for i in range(len(beta_grid)):
# for i in range(6):
  beta = beta_grid[i]
  beta_str = f"{int(beta*1000000):01d}"
  print("beta = ", beta)
  N_spins = 500
  print("N_spins = ", N_spins)
  df = read_df_from_csv('beta_{0}_size_{1}'.format(beta_str,N_spins))
  ec_Classic, ec_Plugin, ec_Adapt, ec_Classic_jack, ec_Plugin_jack, ec_Conservative, \
  ec_Oracle, ec_Onestep, ec_Oracle_jack,ec_Onestep_jack, il_Classic, il_Plugin, il_Adapt, \
  il_Classic_jack, il_Plugin_jack, il_Conservative,il_Simulation, il_Oracle, il_Onestep, \
  il_Oracle_jack, il_Onestep_jack = diagnoistic(df)

  ec_Classic_grid[i] = ec_Classic
  il_Classic_grid[i] = il_Classic
  ec_Plugin_grid[i] = ec_Plugin
  il_Plugin_grid[i] = il_Plugin
  ec_Adapt_grid[i] = ec_Adapt
  il_Adapt_grid[i] = il_Adapt
  ec_Conservative_grid[i] = ec_Conservative
  il_Conservative_grid[i] = il_Conservative
  ec_Classic_jack_grid[i] = ec_Classic_jack
  il_Classic_jack_grid[i] = il_Classic_jack
  ec_Plugin_jack_grid[i] = ec_Plugin_jack
  il_Plugin_jack_grid[i] = il_Plugin_jack
  il_Simulation_grid[i] = il_Simulation
  il_Oracle_jack_grid[i] = il_Oracle_jack
  ec_Oracle_jack_grid[i] = ec_Oracle_jack
  il_Onestep_jack_grid[i] = il_Onestep_jack
  ec_Onestep_jack_grid[i] = ec_Onestep_jack

# Plotting
plt.figure(figsize=(4, 4))

plt.plot(beta_grid, ec_Conservative_grid, marker='o', linestyle='-', label='Conservative', color='blue')
plt.plot(beta_grid, ec_Classic_jack_grid, marker='s', linestyle='--', label=r'$\beta = 0$', color='orange')
plt.plot(beta_grid, ec_Oracle_jack_grid, marker='^', linestyle='-.', label='Oracle', color='green')
plt.plot(beta_grid, ec_Onestep_jack_grid, marker='d', linestyle=':', label='Onestep', color='red')

plt.xlabel(r'$\beta$',fontsize = 14)
plt.ylabel('Empirical Coverage',fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)

plt.show()

# Plotting
plt.figure(figsize=(4, 4))

plt.plot(beta_grid, il_Conservative_grid, marker='o', linestyle='-', label='Conservative', color='blue')
plt.plot(beta_grid, il_Classic_jack_grid, marker='s', linestyle='--', label=r'$\beta = 0$', color='orange')
plt.plot(beta_grid, il_Oracle_jack_grid, marker='^', linestyle='-.', label='Oracle', color='green')
plt.plot(beta_grid, il_Onestep_jack_grid, marker='d', linestyle=':', label='Onestep', color='red')
plt.plot(beta_grid, il_Simulation_grid, marker='x', linestyle='--', label='Simulated', color='cyan')

plt.xlabel(r'$\beta$',fontsize = 14)
plt.ylabel('Interval Length',fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)

plt.show()

"""Plot interval length as a function of sample size."""

il_Conservative_grid = np.zeros(len(beta_grid))
il_Simulation_grid = np.zeros(len(beta_grid))

N_spins_grid = np.array([500, 1000, 2000, 4000, 8000, 16000])

for i in range(len(N_spins_grid)):
  beta = 0
  beta_str = f"{int(beta*1000000):01d}"
  print("beta = ", beta)
  N_spins = N_spins_grid[i]
  print("N_spins = ", N_spins)
  df = read_df_from_csv('beta_{0}_size_{1}'.format(beta_str,N_spins))
  ec_Classic, ec_Plugin, ec_Adapt, ec_Classic_jack, ec_Plugin_jack, ec_Conservative, \
  ec_Oracle, ec_Onestep, ec_Oracle_jack,ec_Onestep_jack,il_Classic, il_Plugin, il_Adapt, \
  il_Classic_jack, il_Plugin_jack, il_Conservative,il_Simulation, il_Oracle, il_Onestep, \
  il_Oracle_jack, il_Onestep_jack = diagnoistic(df)
  il_Conservative_grid[i] = il_Conservative
  il_Simulation_grid[i] = il_Simulation

il_Simulation_grid

il_Conservative_grid

plt.figure(figsize=(4, 4))
plt.plot(np.log10(N_spins_grid), np.log10(il_Conservative_grid), marker='o', linestyle='-')
plt.xlabel(r'$\log_{10}(N_{\mathrm{spins}})$',fontsize=14)
plt.ylabel(r'$\log_{10}(\mathrm{Conservative\ IL})$',fontsize=14)
# plt.title('Log-Conservative IL vs Log-N_spins',fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()

plt.figure(figsize=(4, 4))
plt.plot(np.log10(N_spins_grid), np.log10(il_Simulation_grid), marker='o', linestyle='-')
plt.xlabel(r'$\log_{10}(N_{\mathrm{spins}})$',fontsize=14)
plt.ylabel(r'$\log_{10}(\mathrm{Simulated\ IL})$',fontsize=14)
# plt.title('Log Simulated IL vs Log-N_spins',fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()