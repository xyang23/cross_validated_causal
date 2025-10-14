"""
Simulations varying the number of experimental data for the no-covariate setting. 

Usage: 
    Modify dir_path to save the checkpoint and figures. When running, ignore the SyntaxWarning for figure production.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import json
from datetime import date
import os
from causal_sim import model_class, compute_exp_minmizer, L_exp, L_obs, combined_loss, cross_validation, true_pi_func, tilde_pi_func, lalonde_get_data, generate_data, t_test_normal_baseline

random_seed = 2024
np.random.seed(random_seed)

x_bins = 20 # bins on x axis, 50 in the paper 
max_n_data = 200 # range for observational data size
n_data_vals = np.linspace(5, max_n_data+5, x_bins, dtype=int) # different observational data size
eps = 0.1 # bias
n_sims = 100 # number of simulations, 5000 in the paper
lambda_bin = 5 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
sd_exp = 1 # standard devation of experimental data 
sd_obs = 1 # standard deviation of observational data
true_te = 0.5 # true treatment effect
n_obs = 150 # number of observational data 

# storing results
ours_cv = np.zeros((x_bins, n_sims)) # our method, estimate from cross-validation
exp_only = np.zeros((x_bins, n_sims)) # only using X_exp
obs_only = np.zeros((x_bins, n_sims)) # only using X_obs
# Q_values_all = np.zeros((n_sims, lambda_bin))
lambda_opt_all = np.zeros((x_bins, n_sims)) # lambda values chosen by cross-validation
t_test = np.zeros((x_bins, n_sims)) # T test if two arrays have the same mean, pool if yes

for sim in range(n_sims):
    if sim % 20 == 0:
        print('Simulation', sim)
    # fix the data pool for each simulation
    X_obs = np.random.normal(true_te + eps, sd_obs, size=n_obs)
    X_exp_all =  np.random.normal(true_te, sd_exp, size=max_n_data+5)
    for x_ind in range(x_bins):
        n_data = n_data_vals[x_ind]
        X_exp = np.random.choice(X_exp_all, size=n_data, replace=False) # draw a random subset 
        Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='mean', k_fold=None)
        # Q_values_all[sim] = Q_values
        lambda_opt_all[x_ind][sim] = lambda_opt
        ours_cv[x_ind][sim] =  theta_opt.theta(lambda_opt, X_exp, X_obs)
        exp_only[x_ind][sim] = np.mean(X_exp)
        obs_only[x_ind][sim] =  np.mean(X_obs) 
        t_test[x_ind][sim] = t_test_normal_baseline(x_exp=X_exp, x_obs=X_obs)

# save the checkpoint
data_log = {'Experiment': 'mean_est_N_exp_vs_MSE',
            'Settings': {'x_bins': x_bins, 'n_sims': n_sims, 'lambda_bin': lambda_bin, 'random_seed': random_seed, 
                         'sd_exp': sd_exp, 'sd_obs': sd_obs, 'n_obs': n_obs, 'eps': eps, 'n_exp_range': max_n_data},
            'ours_cv': ours_cv.tolist(),
            'exp_only': exp_only.tolist(),
            'obs_only': obs_only.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
            't_test': t_test.tolist(),
           }
today = str(date.today())
dir_path =  f"./{today}/"
filename = data_log['Experiment'] + '_n_obs_' + str(n_obs) + '_n_exp_range_' + str(data_log['Settings']['n_exp_range']) + '_sd_' + str(sd_exp) + '_eps_' + str(eps) + '_x_bins_' + str(x_bins)
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")
with open(dir_path + filename + '.json', 'w') as f:
    json.dump(data_log, f)
        
"""
Figure 1: MSE comparison (not adjusting the vertical axis).
"""
lambda_opt_mean = np.mean(lambda_opt_all, axis=1)
ours_cv_mean = np.mean((ours_cv - true_te)**2, axis=1)
exp_only_mean = np.mean((exp_only - true_te)**2, axis=1)
obs_only_mean = np.mean((obs_only - true_te)**2, axis=1)
t_test_mean = np.mean((t_test - true_te)**2, axis=1)

filename_fig = filename
plt.rcParams['font.size'] = 10
ax = plt.gca()
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')

markersize = 3.6
plt.plot(n_data_vals, exp_only_mean, color='green', marker='x',  markersize=markersize, label='Only use $X^{\mathrm{exp}}$')
plt.plot(n_data_vals, obs_only_mean, color='brown', marker='v',  markersize=markersize, label='Only use $X^{\mathrm{obs}}$')
plt.plot(n_data_vals, t_test_mean, color='blue', marker='*',  markersize=markersize, label=r'T-test baseline')
plt.plot(n_data_vals, ours_cv_mean, color='orange', marker='.',  markersize=markersize, label=r'Ours, $\beta(\widehat\theta (\widehat\lambda))$')

plt.xlabel('$N^{\mathrm{exp}}$', fontsize=14)
plt.ylabel('Empirical MSE', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(dir_path + filename_fig + '.pdf', format='pdf')
plt.close()

        
"""
Figure 2: MSE comparison (adjusting the vertical aixs).
Current code only produces this figure when both sources have standard deviations 1 or 10. When changing to other cases, one needs to adjust the parameters accordingly.
"""
if (sd_exp == 1 and sd_obs == 1) or (sd_exp == 10 and sd_obs == 10):
    plt.rcParams['font.size'] = 10
    ax = plt.gca()
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    
    a = 5
    y_thresh = np.max(ours_cv_mean)
    def stretch(y, y_thresh=y_thresh, a=a):  # Increase `a` for stronger stretch
        return np.where(y <= y_thresh, a * (y / y_thresh), a + np.log(y / y_thresh))
    
    def inv_stretch(y_trans, y_thresh=y_thresh, a=a):
        return np.where(y_trans <= a, (y_trans / a) * y_thresh, y_thresh * np.exp(y_trans - a))
    
    markersize = 3.6
    plt.plot(n_data_vals, y_thresh*np.ones_like(exp_only_mean), linestyle='--', color='grey')    
    
    plt.xlabel('$N^{\mathrm{exp}}$', fontsize=14)
    plt.ylabel('Empirical MSE', fontsize=14)
    plt.yscale('function', functions=(stretch, inv_stretch))
    
    plt.plot(n_data_vals, exp_only_mean, color='green', marker='x', markersize=markersize, label='Only use $X^{\mathrm{exp}}$')
    plt.plot(n_data_vals, obs_only_mean, color='brown', marker='v', markersize=markersize, label='Only use $X^{\mathrm{obs}}$')
    plt.plot(n_data_vals, t_test_mean, color='blue', marker='*',  markersize=markersize, label=r'T-test baseline')
    plt.plot(n_data_vals, ours_cv_mean, color='orange', marker='.', markersize=markersize, label=r'Ours, $\beta(\widehat\theta (\widehat\lambda))$')
    
    if sd_exp == 1 and sd_obs == 1:
        plt.text(n_data_vals[10], y_thresh+1.9/100, 'Log scale', fontsize=12, color='grey', va='center', ha='left')
        # add text label for the log region (just after the threshold)
        plt.text(n_data_vals[10], y_thresh-0.33/100, 'Linear scale', fontsize=12, color='grey', va='center', ha='left')
        yticks = np.linspace(np.min(ours_cv_mean), y_thresh, 5).tolist() + [.15, .20]
        ylabels = [f"{i:.2f}" for i in (np.array(yticks) * 100).tolist()]      
        plt.text(-15, 0.25, r'$\times 10^{-2}$', fontsize=10, ha='left', va='bottom')
        plt.yticks(yticks, ylabels, fontsize=12)
    elif sd_exp == 10 and sd_obs == 10:
        plt.text(n_data_vals[10], y_thresh+1.9, 'Log scale', fontsize=12, color='grey', va='center', ha='left')
        plt.text(n_data_vals[10], y_thresh-0.33, 'Linear scale', fontsize=12, color='grey', va='center', ha='left')
        yticks = np.linspace(np.min(ours_cv_mean), y_thresh, 5).tolist() + [15, 20]
        plt.yticks(yticks, fontsize=12)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(dir_path + filename_fig + '_adjusted' + '.pdf', format='pdf')
    plt.close()
    
