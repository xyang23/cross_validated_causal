"""
Simulations varying bias for the linear setting. 

Usage: 
    Modify dir_path to save the checkpoint and figures. When running, ignore the SyntaxWarning for figure production. 
    The printed "Simulation" is not in order because of the parallel computing design.
    For parallel computing, change num_workers.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import json
from datetime import date
import os
from causal_sim import model_class, compute_exp_minmizer, L_exp, L_obs, combined_loss, cross_validation, true_pi_func, tilde_pi_func, lalonde_get_data, generate_data
import dask

random_seed = 2024
np.random.seed(random_seed)

x_bins = 20 # bins on x axis, 50 in the paper
eps_range = 0.6 # range for bias
eps_vals = np.linspace(0, eps_range, x_bins) # different bias
n_sims = 100 # number of simulations, 5000 in the paper
lambda_bin = 5 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
n_exp = 50 # number of experimental data 
n_obs =  100 # number of observational data 
d = 5 # dimensions of covariates in both data sources, here we use the same
noise = 1 # standard deviation of noise in outcomes
exp_model ='aipw' # 'aipw', 'response_func', 'mean_diff' - see comments in causal_sim.py
stratified_kfold = True # whether to stratify for cross-validation - see comments in causal_sim.py
true_te = 0.5 # true treatment effect
te_bias = None # bias, to vary
k_fold = 5 # K-fold cross-validation

# storing results
ours_cv = np.zeros((x_bins, n_sims)) # our method, estimate from cross-validation
exp_only = np.zeros((x_bins, n_sims)) # only using X_exp
obs_only = np.zeros((x_bins, n_sims)) # only using X_obs
# Q_values_all = np.zeros((n_sims, lambda_bin))
lambda_opt_all = np.zeros((x_bins, n_sims)) # lambda values chosen by cross-validation

same_cov = False # whether to generate the same true linear model coefficients for both data sources
exp_name = 'linear_eps_vs_MSE'
if same_cov:
    exp_name = exp_name + "_same-cov"
    
data_log = {'Experiment': exp_name,
            'Settings': {'x_bins': x_bins, 'n_sims': n_sims, 'lambda_bin': lambda_bin, 'random_seed': random_seed, 'eps_range': eps_range, 'n_exp': n_exp, 'n_obs': n_obs, 'd': d, 'exp_model': exp_model,'stratified_kfold': stratified_kfold, 'true_te': true_te, 'k_fold': k_fold},
            'ours_cv': ours_cv.tolist(),
            'exp_only': exp_only.tolist(),
            'obs_only': obs_only.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
           }

today = str(date.today())
dir_path =  f"./{today}/"
filename = data_log['Experiment'] + '_eps_range_' + str(data_log['Settings']['eps_range']) + '_n_exp_' + str(n_exp) + '_n_obs_' + str(n_obs) + '_n_sims_' + str(n_sims) 
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")

def run_simulation(sim):
    # one simulation
    print('Simulation', sim)
    res = {} # result for the current run 
    res["lambda_opt_all"] = np.zeros(x_bins)
    res["ours_cv"] = np.zeros(x_bins)
    res["exp_only"] = np.zeros(x_bins)
    res["obs_only"] = np.zeros(x_bins)
    # set random number generator such that each simulation in parallel doesn't generate the same random number for the same seed
    rng = np.random.default_rng(sim)
    true_coef = rng.normal(size=2*d)
    true_coef_exp = true_coef[0:d]
    if same_cov:
        true_coef_obs = true_coef[0:d]
    else:
        true_coef_obs = true_coef[d:2*d]
    # generate experimental data
    Z_exp, A_exp, Y_exp = generate_data(n_exp, d, true_coef_exp, true_te, true_pi_func, noise, rng=rng) 
    for x_ind in range(x_bins):
        te_bias = eps_vals[x_ind]
        # generate observational data (biased)
        Z_obs, A_obs, Y_obs = generate_data(n_obs, d, true_coef_obs, true_te + te_bias, tilde_pi_func, noise)
        X_exp = np.concatenate((Z_exp, A_exp.reshape(-1, 1), Y_exp.reshape(-1, 1)), axis=1)
        X_obs = np.concatenate((Z_obs, A_obs.reshape(-1, 1), Y_obs.reshape(-1, 1)), axis=1)
        Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='linear', 
                                                 k_fold=k_fold, d=d, exp_model=exp_model, stratified_kfold=stratified_kfold, random_state=sim) 
        res["lambda_opt_all"][x_ind] = lambda_opt
        res["ours_cv"][x_ind] =  theta_opt.beta().item()
        exp_mini = compute_exp_minmizer(X_exp, mode='linear', exp_model=exp_model, stratified_kfold=stratified_kfold, d=d)
        res["exp_only"][x_ind] = exp_mini 
        obs_mini = compute_exp_minmizer(X_obs, mode='linear', exp_model=exp_model, stratified_kfold=stratified_kfold, d=d)
        res["obs_only"][x_ind] = obs_mini
    return res
 
if __name__ == '__main__':
    # run simulations in parallel
    dask.config.set(scheduler='processes', num_workers = 1)
    compute_tasks = [dask.delayed(run_simulation)(sim) for sim in range(n_sims)]
    results_list = dask.compute(compute_tasks)[0]
    # organize results
    ours_cv = np.column_stack([res["ours_cv"] for res in results_list])
    lambda_opt_all = np.column_stack([res["lambda_opt_all"] for res in results_list])
    exp_only = np.column_stack([res["exp_only"] for res in results_list])
    obs_only = np.column_stack([res["obs_only"] for res in results_list])
    
    data_log = {'Experiment': exp_name,
                'Settings': {'x_bins': x_bins, 'n_sims': n_sims, 'lambda_bin': lambda_bin, 'random_seed': random_seed, 'eps_range': eps_range, 'n_exp': n_exp, 'n_obs': n_obs, 'd': d, 'exp_model': exp_model,'stratified_kfold': stratified_kfold, 'true_te': true_te, 'k_fold': k_fold},
                'ours_cv': ours_cv.tolist(),
                'exp_only': exp_only.tolist(),
                'obs_only': obs_only.tolist(),
                'lambda_opt_all': lambda_opt_all.tolist(),
               }
    with open(dir_path + filename + '.json', 'w') as f:
        json.dump(data_log, f)
        print('saved file', filename)
    
    """ 
    Figure 1: MSE comparison.
    """
    conf_interval = lambda data: (np.mean(data, axis=1) - np.std(data, axis=1),
                                  np.mean(data, axis=1) +  np.std(data, axis=1))
    
    lambda_opt_mean = np.mean(lambda_opt_all, axis=1)
    ours_cv_mean = np.mean((ours_cv - true_te)**2, axis=1)
    exp_only_mean = np.mean((exp_only - true_te)**2, axis=1)
    obs_only_mean = np.mean((obs_only - true_te)**2, axis=1)
    
    filename_fig = filename
    ax = plt.gca()
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    plt.rcParams['font.size'] = 10
    
    markersize = 3.6
    plt.plot(eps_vals, exp_only_mean, color='green', marker='x',  markersize=markersize, label='Only use $X^{\mathrm{exp}}$')
    plt.plot(eps_vals, obs_only_mean, color='brown', marker='v',  markersize=markersize, label='Only use $X^{\mathrm{obs}}$')
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
    plt.plot(eps_vals, lambda_opt_mean, color='blue', label='Mean of $\widehat\lambda$ selected by cross-validation', marker='x',  markersize=markersize)
    plt.fill_between(eps_vals, *lambda_opt_ci, color='blue', alpha=0.2, label='$\pm$ Standard deviation of $\widehat\lambda$')
    
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
