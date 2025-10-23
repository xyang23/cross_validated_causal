"""
Run our method (linear setting) on the LaLonde dataset with single observational group. 

This can be used to reproduce the intro figure. For full configurations, use lalonde_cv.py or lalonde_cv_bootstrap.

Usage: 
    Modify dir_path to save the checkpoint. 
    Use --group to indicate which observational group to use. Use --variables to indicate covariates. Use --is_bootstrap to enable bootstrap. For example, 
    python lalonde_intro_linear.py --group "psid"
    python lalonde_intro_linear.py --group "psid" --is_bootstrap
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle 
import json
from datetime import date
import os
from causal_sim import model_class, compute_exp_minmizer, L_exp, L_obs, combined_loss, cross_validation, true_pi_func, tilde_pi_func, lalonde_get_data, generate_data

import dask
import pandas as pd
from sklearn.linear_model import LinearRegression 
import argparse

random_seed = 2024
np.random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--group", type=str, default="psid", help="Group name, including: psid, psid2, psid3, cps, cps2, cps3")
parser.add_argument(
    "--variables",
    nargs="*",          
    type=str,
    default=['age', 'education', 'nodegree', 'black', 'hispanic', 'married', 're75', 'u75', 'u74', 're74'],
    help="List of variable names"
)
parser.add_argument("--is_bootstrap",  action="store_true", help="enable bootstrap to calculate standard deviation")

args = parser.parse_args()
group = args.group
variables = args.variables
is_bootstrap = args.is_bootstrap

df = pd.read_csv('lalonde.csv')
df['age2'] = df['age'] ** 2

n_sims = 100 # number of simulations, 5000 in the paper

    
X_exp, X_obs = lalonde_get_data(df, group, variables)

# Experimental data 
# variables = ['age', 'education', 'nodegree', 'black', 'hispanic', 'married', 're75', 'u75', 'u74', 're74']
d = len(variables)
X_exp, X_obs = lalonde_get_data(df, group, variables)

# storing results
ours_cv = np.zeros(( n_sims)) # our method, estimate from cross-validation
exp_only = np.zeros(( n_sims)) # only using X_exp
obs_only = np.zeros(( n_sims)) # only using X_obs

lambda_opt_all = np.zeros(( n_sims)) # lambda values chosen by cross-validation
lambda_bin = 50 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
Q_values_all = np.zeros((n_sims, lambda_bin)) # cross validation objective function

for sim in range(n_sims):
    if sim % 20 == 0:
        print('Simulation', sim)
    rng = np.random.default_rng(sim)
    if is_bootstrap:
        # begin bootstrap
        X_exp_in_use = X_exp[rng.integers(0, X_exp.shape[0], size=X_exp.shape[0])]
        X_obs_in_use = X_obs[rng.integers(0, X_obs.shape[0], size=X_obs.shape[0])]  
    else: 
        X_exp_in_use = X_exp
        X_obs_in_use = X_obs
    Z_exp = X_exp_in_use[:, :d]
    A_exp = X_exp_in_use[:, d]
    Y_exp = X_exp_in_use[:, -1] 
    
    Z_obs = X_obs_in_use[:, :d]
    A_obs = X_obs_in_use[:, d]
    Y_obs = X_obs_in_use[:, -1] 

    exp_model = LinearRegression()
    exp_model.fit(np.concatenate((A_exp.reshape(-1, 1), Z_exp), axis=1), Y_exp)
    exp_estimate = exp_model.coef_[0]
    
    obs_model = LinearRegression()
    obs_model.fit(np.concatenate((A_obs.reshape(-1, 1), Z_obs), axis=1), Y_obs)
    obs_estimate = obs_model.coef_[0]

    Q_values, lambda_opt, theta_opt = cross_validation(X_exp_in_use, X_obs_in_use, lambda_vals, mode='linear', d=d, exp_model='response_func', k_fold=5, random_state=sim)

    lambda_opt_all[sim] = lambda_opt
    ours_cv[sim] =  theta_opt.beta().item()
    exp_only[sim] = exp_estimate
    obs_only[sim] = obs_estimate
    Q_values_all[sim] = Q_values
    
# save the checkpoint
data_log = {'Experiment': 'lalonde_intro_linear',
            'Settings': {'group': group, 'n_sims': n_sims, 'lambda_bin': lambda_bin, 'random_seed': random_seed, 'variables': variables
                      },
            'ours_cv': ours_cv.tolist(),
            'exp_only': exp_only.tolist(),
            'obs_only': obs_only.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
            'Q_values_all': Q_values_all.tolist()
           }

today = str(date.today())
dir_path =  f"./{today}/"
filename = data_log['Experiment'] 
if is_bootstrap:
    filename = filename + '_boot'

filename = filename + '_' + str(data_log['Settings']['group']) + '_' + str(data_log['Settings']['variables'])
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")
with open(dir_path + filename + '.json', 'w') as f:
    json.dump(data_log, f)

# write results

with open(dir_path + filename + ".txt", "w") as f:
    f.write(
        f"exp only, mean and std: {np.mean(exp_only):.1f}  {np.std(exp_only):.1f} \n"
        f"ours, mean and std: {np.mean(ours_cv):.1f}  {np.std(ours_cv):.1f}  \n" 
        f"selected lambda, mean and std: {np.mean(lambda_opt_all)}\n   {lambda_opt_all.std()}\n "
        f"obs only, mean and std: {np.mean(obs_only):.1f}  {np.std(obs_only):.1f} \n"
    )

    f.close()


