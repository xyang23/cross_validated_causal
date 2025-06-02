"""
Run our method on the LaLonde dataset.

Usage: 
    Modify dir_path to save the checkpoint. 
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
import pandas as pd

random_seed = 2024
np.random.seed(random_seed)

variables_list = [[], 
                  ['age', 'age2', 'education','nodegree', 'black', 'hispanic'],
                  ['re75'],
                  ['age', 'age2', 'education', 'nodegree', 'black', 'hispanic', 're75'],
                  ['age', 'education', 'nodegree', 'black', 'hispanic', 'married', 're75', 'u75'],
                  ['age', 'age2', 'education','nodegree', 'black', 'hispanic', 're74'],
                  ['age', 'age2', 'education', 'nodegree', 'black', 'hispanic', 're75', 're74'],
                  ['age', 'education', 'nodegree', 'black', 'hispanic', 'married', 're75', 'u75', 'u74', 're74']
                 ]

group_lists = ['psid', 'psid2', 'psid3', 'cps', 'cps2', 'cps3'] # could also be 'control'

df = pd.read_csv('lalonde.csv')
df['age2'] = df['age'] ** 2

lambda_bin = 5 # number of candidate values of lambda, 50 in the paper
n_sims = 100 # number of simulations, 5000 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
exp_model = 'response_func' # plug-in estimator 
stratified_kfold = True # whether to stratify for cross-validation - see comments in causal_sim.py
k_fold = 5 # K-fold cross-validation

# storing results
ours_cv = np.zeros((len(group_lists), len(variables_list))) # our method, estimate from cross-validation
lambda_opt_all = np.zeros((len(group_lists), len(variables_list))) # lambda values chosen by cross-validation

exp_name = 'lalonde_cv'

# save the checkpoint
data_log = {'Experiment': exp_name,
            'Settings': {'lambda_bin': lambda_bin, 'random_seed': random_seed, 'n_sims': n_sims,
                        'stratified_kfold': stratified_kfold, 'k_fold': k_fold},
            'ours_cv': ours_cv.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
           }
today = str(date.today())
dir_path =  f"./{today}/" 
filename = data_log['Experiment'] + '_n_sims_' + str(n_sims) + '_lambda_bin_' + str(data_log['Settings']['lambda_bin']) 
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")
       
def run_simulation(sim):
    print('Simulation', sim)
    res = {} # result for the current run 
    
    res["lambda_opt_all"] = np.zeros((len(group_lists), len(variables_list)))
    res["ours_cv"] = np.zeros((len(group_lists), len(variables_list)))

    for group_id, group in enumerate(group_lists):
        for variables_id, variables in enumerate(variables_list):
            d = len(variables)
            X_exp, X_obs = lalonde_get_data(df, group, variables)
            Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='linear', 
                                                     k_fold=k_fold, d=d, exp_model=exp_model, stratified_kfold=stratified_kfold, random_state=sim)                                    
            lambda_opt_all[group_id][variables_id] = lambda_opt
            ours_cv[group_id][variables_id] =  theta_opt.beta().item()
            res["lambda_opt_all"][group_id][variables_id] = lambda_opt
            res["ours_cv"][group_id][variables_id] =  theta_opt.beta().item()
    return res
  
    
if __name__ == '__main__':
    # run simulations in parallel
    dask.config.set(scheduler='processes', num_workers = 1)
    compute_tasks = [dask.delayed(run_simulation)(sim) for sim in range(n_sims)]
    results_list = dask.compute(compute_tasks)[0]

    ours_cv = np.stack([res["ours_cv"] for res in results_list], axis=0)
    lambda_opt_all = np.stack([res["lambda_opt_all"] for res in results_list], axis=0)
    data_log = {'Experiment': exp_name,
            'Settings': {'lambda_bin': lambda_bin, 'random_seed': random_seed, 'n_sims': n_sims,
                        'stratified_kfold': stratified_kfold, 'k_fold': k_fold},
            'ours_cv': ours_cv.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
           }

    with open(dir_path + filename + '.json', 'w') as f:
        json.dump(data_log, f)
        print('saved file', filename)

    """
    Produce a table of results.
    """
    lambda_mean = np.mean(lambda_opt_all, axis=0)
    lambda_std = np.std(lambda_opt_all, axis=0)
    theta_mean = np.mean(ours_cv, axis=0)
    theta_std = np.std(ours_cv, axis=0)
    latex_table = ""
    
    def floor_str(x):
        # floor + 0.5 and convert to str
        return str(int(np.floor(x + 0.5)))
    
    for group_id in range(len(group_lists)):
        cur_latex_lambda = ""
        cur_latex_theta = ""
        for variables_id in range(len(variables_list)):
            cur_latex_theta +=  " & " +  floor_str(theta_mean[group_id][variables_id]) + "$\\pm$" + floor_str(theta_std[group_id][variables_id]) 
            cur_latex_lambda +=  " & (" + f"{lambda_mean[group_id][variables_id]:.1f}"  + "$\\pm$" + f"{lambda_std[group_id][variables_id]:.1f}"  + ")"   
        latex_table += group_lists[group_id] +  cur_latex_theta + " \\\\\n"
        latex_table += cur_latex_lambda + " \\\\\n"
    print(latex_table)
    with open(dir_path + filename + '.txt', 'w') as f:
        f.write(latex_table)
        f.close()