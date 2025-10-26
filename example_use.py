"""
Example use of our method.
"""

import numpy as np
from causal_sim import cross_validation, generate_data, true_pi_func, tilde_pi_func
print('--------------------------------------------------------------------------------')
print('Running')
print('--------------------------------------------------------------------------------')


"""
Example use of the no-covariate setting.
"""
n_exp = 100 # number of experimental data 
n_obs = 200 # number of observational data 
sd_exp = 1 # standard devation of experimental data 
sd_obs = 1 # standard deviation of observational data
true_te = 0.5 # true treatment effect
eps = 0.05 # bias 
lambda_bin = 5 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values

X_exp = np.random.normal(true_te, sd_exp, size=n_exp)
X_obs = np.random.normal(true_te + eps, sd_obs, size=n_obs)
#Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='mean', k_fold=None) # leave-one-out cross-validation
Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='mean', k_fold=5,  random_state=2024) # five cross-validation

estimate = theta_opt.theta(lambda_opt, X_exp, X_obs)
print('--------------------------------------------------------------------------------')
print('No-covariate setting')
print('True treatment effect:', true_te)
print(f"Our estimate is {estimate:.3f}, with lambda = {lambda_opt:.2f} selected by cross-validation" )
print('--------------------------------------------------------------------------------')

"""
Example use of the linear setting.
"""
n_exp = 1000 # number of experimental data 
n_obs = 2000 # number of observational data 
noise = 1 # standard deviation of noise in outcomes
true_te = 0.5 # true treatment effect
eps = 0.05 # bias 
lambda_bin = 5 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
k_fold = 5 # K-fold cross-validation
exp_model ='response_func' # 'aipw', 'response_func', 'mean_diff' - see comments in causal_sim.py
d_exp = 5 # dimensions of covariates in experimental data
d_obs = 10 # dimensions of covariates in observational data
true_coef = np.random.normal(size=d_exp+d_obs) # linear model coefficients to generate data
true_coef_exp = true_coef[0:d_exp]
true_coef_obs = true_coef[d_exp:d_exp+d_obs]
Z_exp, A_exp, Y_exp = generate_data(n_exp, d_exp, true_coef_exp, true_te, true_pi_func, noise) 
Z_obs, A_obs, Y_obs = generate_data(n_obs, d_obs, true_coef_obs, true_te + eps, tilde_pi_func, noise)

X_exp = np.concatenate((Z_exp, A_exp.reshape(-1, 1), Y_exp.reshape(-1, 1)), axis=1) # structured as: covariates, treatment, outcome
X_obs = np.concatenate((Z_obs, A_obs.reshape(-1, 1), Y_obs.reshape(-1, 1)), axis=1)
Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='linear', k_fold=k_fold, d_exp=d_exp, d_obs=d_obs, exp_model=exp_model, random_state=2024)
estimate = theta_opt.beta().item()

print('--------------------------------------------------------------------------------')
print('Linear setting')
print('True treatment effect:', true_te)
print(f"Our estimate is {estimate:.3f}, with lambda = {lambda_opt:.2f} selected by cross-validation" )
print('--------------------------------------------------------------------------------')
