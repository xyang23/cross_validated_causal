"""
Simulations varying bias for the no-covariate setting. 

Usage: 
    Modify dir_path to save the checkpoint and figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import json
from datetime import date
import os
from causal_sim import cross_validation, t_test_normal_baseline

random_seed = 2024
np.random.seed(random_seed)

        
x_bins = 20 # bins on x axis, 50 in the paper
eps_range = 0.2 # range for bias
eps_vals = np.linspace(0, eps_range, x_bins) # different bias
n_sims = 100 # number of simulations, 5000 in the paper
lambda_bin = 5 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
n_exp = 100 # number of experimental data 
n_obs = 200 # number of observational data 
sd_exp = 1 # standard devation of experimental data 
sd_obs = 1 # standard deviation of observational data
true_te = 0.5 # true treatment effect

# storing results
ours_cv = np.zeros((x_bins, n_sims)) # our method, estimate from cross-validation
exp_only = np.zeros((x_bins, n_sims)) # only using X_exp
obs_only = np.zeros((x_bins, n_sims)) # only using X_obs
# Q_values_all = np.zeros((n_sims, lambda_bin))
lambda_opt_all = np.zeros((x_bins, n_sims)) # lambda values chosen by cross-validation
t_test = np.zeros((x_bins, n_sims)) # T test if two arrays have the same mean, pool if yes. 

for sim in range(n_sims):
    X_exp = np.random.normal(true_te, sd_exp, size=n_exp)
    if sim % 20 == 0:
        print('Simulation', sim)
    for x_ind in range(x_bins):
        eps = eps_vals[x_ind]
        X_obs = np.random.normal(true_te + eps, sd_obs, size=n_obs)
        Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='mean', k_fold=None)
        # Q_values_all[sim] = Q_values
        lambda_opt_all[x_ind][sim] = lambda_opt
        ours_cv[x_ind][sim] =  theta_opt.theta(lambda_opt, X_exp, X_obs)
        exp_only[x_ind][sim] = np.mean(X_exp)
        obs_only[x_ind][sim] = np.mean(X_obs)
        t_test[x_ind][sim] = t_test_normal_baseline(x_exp=X_exp, x_obs=X_obs)
    
# save the checkpoint
data_log = {'Experiment': 'mean_est_eps_t_test',
            'Settings': {'x_bins': x_bins, 'n_sims': n_sims, 'lambda_bin': lambda_bin, 'random_seed': random_seed, 
                         'sd_exp': sd_exp, 'sd_obs': sd_obs, 'eps_range': eps_range, 'n_exp': n_exp, 'n_obs': n_obs},
            'ours_cv': ours_cv.tolist(),
            'exp_only': exp_only.tolist(),
            'obs_only': obs_only.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
            't_test': t_test.tolist(),
           }
today = str(date.today())
dir_path =  f"./{today}/"
filename = data_log['Experiment'] + '_eps_range_' + str(data_log['Settings']['eps_range']) + '_sd_' + str(sd_exp) + '_n_exp_' + str(n_exp) +'_n_obs_' + str(n_obs) + '_n_sims_' + str(n_sims)
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")
    
with open(dir_path + filename + '.json', 'w') as f:
    json.dump(data_log, f)

""" 
Figure 1: MSE comparison.
"""
conf_interval = lambda data: (np.mean(data, axis=1) - np.std(data, axis=1),
                              np.mean(data, axis=1) +  np.std(data, axis=1))

lambda_opt_mean = np.mean(lambda_opt_all, axis=1)
ours_cv_mean = np.mean((ours_cv - true_te)**2, axis=1)
exp_only_mean = np.mean((exp_only - true_te)**2, axis=1)
obs_only_mean = np.mean((obs_only - true_te)**2, axis=1)
t_test_mean = np.mean((t_test - true_te)**2, axis=1)

filename_fig = filename
ax = plt.gca()
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
plt.rcParams['font.size'] = 10

markersize = 3.6
plt.plot(eps_vals, exp_only_mean, color='green', marker='x',  markersize=markersize, label=r'Only use $X^{\mathrm{exp}}$')
plt.plot(eps_vals, obs_only_mean, color='brown', marker='v',  markersize=markersize, label=r'Only use $X^{\mathrm{obs}}$')
plt.plot(eps_vals, t_test_mean, color='blue', marker='*',  markersize=markersize, label=r'T-test baseline')
plt.plot(eps_vals, ours_cv_mean, color='orange', marker='.',  markersize=markersize, label=r'Ours, $\beta(\widehat\theta (\widehat\lambda))$')


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$\varepsilon$', fontsize=14)
plt.ylabel('Empirical MSE', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(dir_path + filename_fig + '.pdf', format='pdf')
plt.close()

""" 
Figure 2: MSE ratio and selected lambda.
""" 
filename_fig = filename + '_MSE_ratio'
ax = plt.gca()
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
plt.rcParams['font.size'] = 10

markersize = 5
lambda_opt_ci = conf_interval(lambda_opt_all)
plt.plot(eps_vals, lambda_opt_mean, color='blue', label=r'Mean of $\widehat\lambda$ selected by cross-validation', marker='x',  markersize=markersize)
plt.fill_between(eps_vals, *lambda_opt_ci, color='blue', alpha=0.2, label=r'$\pm$ Standard deviation of $\widehat\lambda$')

exp_ours_ratio = np.divide(exp_only_mean, ours_cv_mean)

plt.plot(eps_vals, exp_ours_ratio, color='lightseagreen', marker='v',  markersize=markersize, label=r'MSE ratio between only using ${X}^{\mathrm{exp}}$ and ours')
plt.plot(eps_vals, np.ones_like(exp_ours_ratio), linestyle='--', color='grey')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$\varepsilon$', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(dir_path + filename_fig + '.pdf', format='pdf')
plt.close()