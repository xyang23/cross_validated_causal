"""
Run our method (no-covariate setting) on the LaLonde dataset with single observational group. 

This can be used to reproduce the intro figure. 

Usage: 
    Modify dir_path to save the checkpoint. 
    Use --group to indicate which observational group to use. Use --is_bootstrap to enable bootstrap. For example, 
    python lalonde_intro_mean.py --group "psid"
    python lalonde_intro_mean.py --group "psid" --is_bootstrap
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle 
import json
from datetime import date
import os
from causal_sim import cross_validation, lalonde_get_data, t_test_normal_baseline 
import dask
import pandas as pd
import argparse


random_seed = 2024
np.random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--group", type=str, default="psid", help="Group name, including: psid, psid2, psid3, cps, cps2, cps3 ")
parser.add_argument("--is_bootstrap",  action="store_true", help="enable bootstrap to calculate standard deviations")

args = parser.parse_args()
group = args.group
is_bootstrap = args.is_bootstrap

df = pd.read_csv('lalonde.csv')
df['age2'] = df['age'] ** 2

n_sims = 100 # number of simulations, 5000 in the paper

variables = []  
X_exp, X_obs = lalonde_get_data(df, group, variables)

# Experimental data control

X_exp_real_control = X_exp[X_exp[:, 0] == 0]
X_exp_real_control = X_exp_real_control[:, 1]

X_exp_real_treated = X_exp[X_exp[:, 0] == 1]
X_exp_real_treated = X_exp_real_treated[:, 1]
treated_mean = np.mean(X_exp_real_treated)

# Observational data control
X_obs_real_control = X_obs[X_obs[:, 0] == 0]
X_obs_real_control = X_obs_real_control[:, 1]


ours_cv = np.zeros((n_sims)) # our method, estimate from cross-validation
exp_only = np.zeros((n_sims)) # only using X_exp
obs_only = np.zeros((n_sims)) # only using X_obs

lambda_opt_all = np.zeros((n_sims)) # lambda values chosen by cross-validation
t_test = np.zeros((n_sims)) # T test if two arrays have the same mean, pool if yes. 
lambda_bin = 50 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
Q_values_all = np.zeros((n_sims, lambda_bin)) # cross validation objective function

for sim in range(n_sims):
    if sim % 20 == 0:
        print('Simulation', sim)
    rng = np.random.default_rng(sim)
    if is_bootstrap:
        X_exp_control_in_use = X_exp_real_control[rng.integers(0, X_exp_real_control.shape[0], size=X_exp_real_control.shape[0])]
        X_obs_control_in_use = X_obs_real_control[rng.integers(0, X_obs_real_control.shape[0], size=X_obs_real_control.shape[0])]
    else:
        X_exp_control_in_use = X_exp_real_control
        X_obs_control_in_use = X_obs_real_control

    Q_values, lambda_opt, theta_opt = cross_validation(X_exp_control_in_use, X_obs_control_in_use, lambda_vals, mode='mean', k_fold=5)
    Q_values_all[sim] = Q_values
    
    lambda_opt_all[sim] = lambda_opt
    # ATE = treated mean - control mean
    ours_cv[sim] = treated_mean - theta_opt.theta(lambda_opt, X_exp_control_in_use , X_obs_control_in_use)
    exp_only[sim] = treated_mean - np.mean(X_exp_control_in_use)
    obs_only[sim] = treated_mean - np.mean(X_obs_control_in_use)
    t_test[sim] = treated_mean - t_test_normal_baseline(x_exp=X_exp_control_in_use , x_obs=X_obs_control_in_use , equal_var=False)

# save the checkpoint
data_log = {'Experiment': 'lalonde_intro_mean',
            'Settings': {'group': group, 'n_sims': n_sims, 'lambda_bin': lambda_bin, 'random_seed': random_seed, 'treated_mean': treated_mean},
            'ours_cv': ours_cv.tolist(),
            'exp_only': exp_only.tolist(),
            'obs_only': obs_only.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
            't_test': t_test.tolist(),
            'Q_values_all': Q_values_all.tolist(),
           }
today = str(date.today())
dir_path =  f"./{today}/"
filename = data_log['Experiment'] + '_' + str(data_log['Settings']['group'])
if is_bootstrap:
    filename = filename + '_boot'
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")
with open(dir_path + filename + '.json', 'w') as f:
    json.dump(data_log, f)

with open(dir_path + filename + ".txt", "w") as f:
    f.write(
        f"Treated mean: {treated_mean:.1f}\n"
        f"Treated mean - Control mean: \n"
        f"exp only, mean and std: {np.mean(exp_only):.1f}  {np.std(exp_only):.1f} \n"
        f"ours, mean and std: {np.mean(ours_cv):.1f}  {np.std(ours_cv):.1f}  \n" 
        f"selected lambda, mean and std: {np.mean(lambda_opt_all)}\n   {lambda_opt_all.std()}\n "
        f"obs only, mean and std: {np.mean(obs_only):.1f}  {np.std(obs_only):.1f} \n"
    )
    f.close()


