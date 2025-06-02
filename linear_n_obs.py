"""
Simulations varying the number of observational data for the linear setting. 

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
max_n_data = 200 # range for observational data size
n_data_vals = np.linspace(20, max_n_data, x_bins, dtype=int) # reason to start from 20 is that too few data could result in a singular matrix when solving the linear system
n_exp = 50 # number of experimental data 
n_obs = None # number of observational data, to vary
n_sims = 100 # number of simulations, 5000 in the paper
lambda_bin = 5 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
true_te = 0.5 # true treatment effect
te_bias = 0.05 # bias 
k_fold = 5 # K-fold cross-validation
d = 5 # dimensions of covariates in both data sources, here we use the same
noise = 1 # standard deviation of noise in outcomes
exp_model = 'aipw' # 'aipw', 'response_fuc', 'mean_diff' - see comments in causal_sim.py
stratified_kfold = True # whether to stratify for cross-validation - see comments in causal_sim.py

# storing results
ours_cv = np.zeros((x_bins, n_sims)) # our method, estimate from cross-validation
exp_only = np.zeros((x_bins, n_sims)) # only using X_exp
obs_only = np.zeros((x_bins, n_sims)) # only using X_obs
# Q_values_all = np.zeros((n_sims, lambda_bin))
lambda_opt_all = np.zeros((x_bins, n_sims))

same_cov = False # whether to generate the same true linear model coefficients for both data sources
exp_name = 'linear_N_obs_vs_MSE'
if same_cov:
    exp_name = exp_name + "_same-cov"
true_coef = np.random.randn(2*d)
true_coef_exp = true_coef[0:d]
if same_cov:
    true_coef_obs = true_coef[0:d]
else:
    true_coef_obs = true_coef[d:2*d] 
 
 
data_log = {'Experiment': exp_name,
        'Settings': {'x_bins': x_bins, 'n_sims': n_sims, 'lambda_bin': lambda_bin, 'random_seed': random_seed, 'max_n_data': max_n_data, 'n_exp': n_exp, 'te_bias': te_bias, 'd': d, 'exp_model': exp_model, 'stratified_kfold': stratified_kfold, 'true_te': true_te, 'k_fold': k_fold, 'true_coef': true_coef.tolist()}, 
        'ours_cv': ours_cv.tolist(),
        'exp_only': exp_only.tolist(),
        'obs_only': obs_only.tolist(),
        'lambda_opt_all': lambda_opt_all.tolist(),
       }  

today = str(date.today())
dir_path =  f"./{today}/" 

filename = data_log['Experiment'] + '_max_n_data_' + str(data_log['Settings']['max_n_data']) + '_n_exp_' + str(n_exp) + '_te_bias_' + str(te_bias) + '_n_sims_' + str(n_sims) 

print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")
      
        
# for parallel computing
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
    Z_exp, A_exp, Y_exp = generate_data(n_exp, d, true_coef_exp, true_te, true_pi_func, noise, rng=rng) 
    Z_obs_all, A_obs_all, Y_obs_all = generate_data(max_n_data, d, true_coef_obs, true_te + te_bias, tilde_pi_func, noise, rng=rng) 
    X_exp = np.concatenate((Z_exp, A_exp.reshape(-1, 1), Y_exp.reshape(-1, 1)), axis=1)
    X_obs_all = np.concatenate((Z_obs_all, A_obs_all.reshape(-1, 1), Y_obs_all.reshape(-1, 1)), axis=1)
    
    for x_ind in range(x_bins):
        n_data = n_data_vals[x_ind]
        indices = rng.choice(X_obs_all.shape[0], size=n_data, replace=False)
        X_obs = X_obs_all[indices]
        Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='linear', 
                                                 k_fold=k_fold, d=d, exp_model=exp_model, stratified_kfold=stratified_kfold, random_state=sim)
        exp_mini = compute_exp_minmizer(X_exp, mode='linear', exp_model=exp_model, stratified_kfold=stratified_kfold, d=d)
        obs_mini = compute_exp_minmizer(X_obs, mode='linear', exp_model=exp_model, stratified_kfold=stratified_kfold, d=d)
        res["lambda_opt_all"][x_ind] = lambda_opt
        res["ours_cv"][x_ind] =  theta_opt.beta().item()
        res["exp_only"][x_ind] = exp_mini 
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
        'Settings': {'x_bins': x_bins, 'n_sims': n_sims, 'lambda_bin': lambda_bin, 'random_seed': random_seed, 'max_n_data': max_n_data, 'n_exp': n_exp, 'te_bias': te_bias, 'd': d, 'exp_model': exp_model, 'stratified_kfold': stratified_kfold, 'true_te': true_te, 'k_fold': k_fold, 'true_coef': true_coef.tolist(),}, 
        'ours_cv': ours_cv.tolist(),
        'exp_only': exp_only.tolist(),
        'obs_only': obs_only.tolist(),
        'lambda_opt_all': lambda_opt_all.tolist(),
       }  
    with open(dir_path + filename + '.json', 'w') as f:
        json.dump(data_log, f)
        print('saved file', filename)
      
            
    """
    Figure 1: MSE comparison (not adjusting the vertical axis).
    """
    lambda_opt_mean = np.mean(lambda_opt_all, axis=1)
    ours_cv_mean = np.mean((ours_cv - true_te)**2, axis=1)
    exp_only_mean = np.mean((exp_only - true_te)**2, axis=1)
    obs_only_mean = np.mean((obs_only - true_te)**2, axis=1)
    
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
    plt.plot(n_data_vals, ours_cv_mean, color='orange', marker='.',  markersize=markersize, label=r'Ours, $\beta(\widehat\theta (\widehat\lambda))$')
    
    plt.xlabel('$N^{\mathrm{obs}}$', fontsize=14)
    plt.ylabel('Empirical MSE', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(dir_path + filename_fig + '.pdf', format='pdf')
    plt.close()
             
    """
    Figure 2: MSE comparison (adjusting the vertical aixs).
    Current code only produces this for specific settings. When changing to other cases, one needs to adjust the parameters accordingly.
    """
    if (n_exp == 50 and max_n_data == 200) or (n_exp == 1000 and max_n_data == 2000):
        
        plt.rcParams['font.size'] = 10 
        ax = plt.gca()
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        
        y_thresh = np.max(ours_cv_mean)
        if n_exp == 50 and max_n_data == 200:  
            a = 3
        elif n_exp == 1000 and max_n_data == 2000:  
            a = 5
            
        def stretch(y, y_thresh=y_thresh, a=a):  # Increase `a` for stronger stretch
            return np.where(y <= y_thresh, a * (y / y_thresh), a + np.log(y / y_thresh))
        
        def inv_stretch(y_trans, y_thresh=y_thresh, a=a):
            return np.where(y_trans <= a, (y_trans / a) * y_thresh, y_thresh * np.exp(y_trans - a))
        
        markersize = 3.6
        plt.plot(n_data_vals, y_thresh*np.ones_like(exp_only_mean), linestyle='--', color='grey')
        
        if n_exp == 50 and max_n_data == 200:  
            plt.text(n_data_vals[10], y_thresh+0.03, 'Log scale', fontsize=12, color='grey', va='center', ha='left')
            plt.text(n_data_vals[10], y_thresh-0.01, 'Linear scale', fontsize=12, color='grey', va='center', ha='left')
            yticks = np.linspace(np.min(ours_cv_mean), y_thresh, 5).tolist() + [.5, 1, 1.5, 2, 2.5, 3]
        elif n_exp == 1000 and max_n_data == 2000:  
            a = 5
            plt.text(n_data_vals[10], y_thresh+0.0021, 'Log scale', fontsize=12, color='grey', va='center', ha='left')
            plt.text(n_data_vals[10], y_thresh-0.00041, 'Linear scale', fontsize=12, color='grey', va='center', ha='left')
            yticks = np.linspace(np.min(ours_cv_mean), y_thresh, 5).tolist() + [.05, .1, .5, 1, 2, 3]
  
        plt.xlabel('$N^{\mathrm{obs}}$', fontsize=14)
        plt.ylabel('Empirical MSE', fontsize=14)
        plt.yscale('function', functions=(stretch, inv_stretch))
        
        plt.plot(n_data_vals, exp_only_mean, color='green', marker='x', markersize=markersize, label='Only use $X^{\mathrm{exp}}$')
        plt.plot(n_data_vals, obs_only_mean, color='brown', marker='v', markersize=markersize, label='Only use $X^{\mathrm{obs}}$')
        plt.plot(n_data_vals, ours_cv_mean, color='orange', marker='.', markersize=markersize, label=r'Ours, $\beta(\widehat\theta (\widehat\lambda))$')
        
        plt.yticks(yticks, fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        plt.savefig(dir_path + filename_fig + '_adjusted' + '.pdf', format='pdf')
        plt.close()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
 