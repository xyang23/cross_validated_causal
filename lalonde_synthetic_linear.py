"""
Synthetic data based on the LaLonde dataset (linear setting) with single observational group. 


Usage: 
    Modify dir_path to save the checkpoint. 
    Use --group to indicate which observational group to use. Use --variables to indicate covariates. For example, 
    python lalonde_synthetic_linear.py --group "psid"
    python lalonde_synthetic_linear.py --group 'psid' --variables 're75'
    python lalonde_synthetic_linear.py --group 'psid' --variables 'age' 'education' 'nodegree' 'black' 'hispanic' 'married' 're75' 'u75' 'u74' 're74' 
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle 
import json
from datetime import date
import os
from causal_sim import model_class, compute_exp_minmizer, L_exp, L_obs, combined_loss, cross_validation, true_pi_func, tilde_pi_func, lalonde_get_data, generate_data
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
    default=[],
    help="List of variable names"
)
# column 3 variables: ['re75']
# column 8 variables: ['age', 'education', 'nodegree', 'black', 'hispanic', 'married', 're75', 'u75', 'u74', 're74']

args = parser.parse_args()
group = args.group
variables = args.variables

df = pd.read_csv('lalonde.csv')
df['age2'] = df['age'] ** 2

n_sims = 100 # number of simulations, 5000 in the paper
 
d = len(variables)
X_exp, X_obs = lalonde_get_data(df, group, variables)

# Experimental data 
Z_exp = X_exp[:, :d]
A_exp = X_exp[:, d]
Y_exp_real = X_exp[:, -1] 
# fit a linear model, then get residuals mean and std
exp_model = LinearRegression()
exp_model.fit(np.concatenate((A_exp.reshape(-1, 1), Z_exp), axis=1), Y_exp_real)
Y_predicted = exp_model.predict(np.concatenate((A_exp.reshape(-1, 1), Z_exp),axis=1))
residuals = Y_exp_real - Y_predicted
residuals_std = np.sqrt(residuals.var(ddof=1))
n_exp = X_exp.shape[0]
true_te = exp_model.coef_[0]

# Observational data
Z_obs = X_obs[:, :d]
A_obs = X_obs[:, d]
Y_obs = X_obs[:, -1] 
# fit a linear model, then get residuals mean and std
obs_model = LinearRegression()
obs_model.fit(np.concatenate((A_obs.reshape(-1, 1), Z_obs), axis=1), Y_obs)

Y_obs_predicted = obs_model.predict(np.concatenate((A_obs.reshape(-1, 1), Z_obs),axis=1))
obs_residuals = Y_obs - Y_obs_predicted
obs_residuals_std = np.sqrt(obs_residuals.var(ddof=1))
n_obs = X_obs.shape[0]

# storing results
ours_cv = np.zeros(( n_sims)) # our method, estimate from cross-validation
exp_only = np.zeros(( n_sims)) # only using X_exp
obs_only = np.zeros(( n_sims)) # only using X_obs
# Q_values_all = np.zeros((n_sims, lambda_bin))
lambda_opt_all = np.zeros(( n_sims)) # lambda values chosen by cross-validation
lambda_bin = 50 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values

for sim in range(n_sims):
    if sim % 20 == 0:
        print('Simulation', sim)
    rng = np.random.default_rng(sim)
    residuals_sim = rng.normal(0, residuals_std, size=n_exp)
    Y_exp = Y_predicted + residuals_sim
    X_exp =  np.concatenate((Z_exp, A_exp.reshape(-1, 1), Y_exp.reshape(-1, 1)), axis=1)

    exp_model = LinearRegression()
    exp_model.fit(np.concatenate((A_exp.reshape(-1, 1), Z_exp), axis=1), Y_exp)
    exp_estimate = exp_model.coef_[0]
    
    # use gaussian obs
    obs_residuals_sim = rng.normal(0, obs_residuals_std, size=n_obs)
    Y_obs = Y_obs_predicted + obs_residuals_sim
    X_obs =  np.concatenate((Z_obs, A_obs.reshape(-1, 1), Y_obs.reshape(-1, 1)), axis=1)
    obs_model = LinearRegression()
    obs_model.fit(np.concatenate((A_obs.reshape(-1, 1), Z_obs), axis=1), Y_obs)
    obs_estimate = obs_model.coef_[0]

    Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='linear', d=d, exp_model='response_func', k_fold=5, random_state=sim)

    lambda_opt_all[sim] = lambda_opt
    ours_cv[sim] =  theta_opt.beta().item()
    exp_only[sim] = exp_estimate
    obs_only[sim] = obs_estimate


# save the checkpoint
data_log = {'Experiment': 'lalonde_synthetic_linear',
            'Settings': {'group': group, 'n_sims': n_sims, 'lambda_bin': lambda_bin, 'random_seed': random_seed, 'true_te': true_te, 'variables': variables
                      },
            'ours_cv': ours_cv.tolist(),
            'exp_only': exp_only.tolist(),
            'obs_only': obs_only.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
           }
today = str(date.today())
dir_path =  f"./{today}/"
filename = data_log['Experiment'] 
filename = filename + '_' + str(data_log['Settings']['group']) + '_' + str(data_log['Settings']['variables'])
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")
with open(dir_path + filename + '.json', 'w') as f:
    json.dump(data_log, f)

# write results
lambda_opt_mean = np.mean(lambda_opt_all)
ours_cv_mean = np.mean((ours_cv - true_te)**2)
exp_only_mean = np.mean((exp_only - true_te)**2)
obs_only_mean = np.mean((obs_only - true_te)**2)

with open(dir_path + filename + ".txt", "w") as f:
    # f.write(
    #     f"MSE:\n"
    #     f"ours: {ours_cv_mean}\n"
    #     f"exp only: {exp_only_mean}\n"
    #     f"obs only: {obs_only_mean}\n"
    # )
    f.write(
        f"selected lambda mean: {lambda_opt_mean}\n"
        f"selected lambda sd: {lambda_opt_all.std()}\n"
    )
    f.write(
        f"RMSE\n"
        f"ours: {np.sqrt(ours_cv_mean)}\n"
        f"exp only: {np.sqrt(exp_only_mean)}\n"
        f"obs only: {np.sqrt(obs_only_mean)}\n"
    )
    f.close()


