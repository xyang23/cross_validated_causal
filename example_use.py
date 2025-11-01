"""
Example use of our method.
"""

import numpy as np
from causal_sim import model_class, cross_validation, generate_data, true_pi_func
import matplotlib.pyplot as plt
from datetime import date
import os


# create a folder to save results
today = str(date.today())
dir_path =  f"./{today}/" 
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created to save the figures.")


"""
Example 1.1

No-covariate setting. Estimate the control group's mean given a *specific* lambda. Treated group proceeds analogously. 

"""
treated_mean = 1 # treated group's mean, assuming experimental and observational data sharing the same treated group (similar to the LaLonde's data)
lambda_ = 0.7 # set a specific lambda
true_mean = 0.5 # true mean
eps = 0.05 # bias in observational data

X_exp = np.random.normal(true_mean, 1, size=100) 
X_obs = np.random.normal(true_mean + eps, 1, size=200)
model = model_class(mode='mean')
model.fit_model(lambda_, X_exp, X_obs) 
estimate_given_lambda = model.beta(lambda_, X_exp, X_obs)
print('--------------------------------------------------------------------------------')
print('No-covariate setting')
print('True control mean:', true_mean)
print(f"Given lambda={lambda_}, for example, estimated control mean = {estimate_given_lambda:.3f}", )
print('Sanity check: (1 - lambda) * mean of exp sample + lambda * mean of obs sample =', 
      f"{(1-lambda_)*np.mean(X_exp)+lambda_*np.mean(X_obs) :.3f}")
print(f"Given lambda={lambda_}, treatment effect = treated mean ({treated_mean}) - estimated control mean = {treated_mean -  estimate_given_lambda :.3f}" )

"""
Example 1.2

No-covariate setting. Using cross-validation to select a lambda.

"""
lambda_bin = 10 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values

#Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='mean', k_fold=None) # leave-one-out cross-validation
Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='mean', k_fold=5, random_state=2024) # five cross-validation
estimate = theta_opt.theta(lambda_opt, X_exp, X_obs)

print(f"\nBy cross-validation, our method selects lambda = {lambda_opt:.2f}" )
print(f"Given lambda={lambda_opt:.2f}, our estimate is {estimate:.3f}")
print(f"Given lambda={lambda_opt:.2f}, treatment effect = treated mean ({treated_mean}) - estimated control mean = {treated_mean -  estimate :.3f}" )

# plot the cross-validation objective function
plt.figure()
plt.title('Cross-validation objective')
plt.plot(lambda_vals, np.log(Q_values), color='black')
plt.xlabel(r'$\lambda$',)
plt.ylabel(r'log CV($\lambda$)')
plt.tight_layout()
filename_fig="example_use_no_covariate"
plt.savefig(dir_path + filename_fig + '.pdf', format='pdf')
plt.close()
print('figure saved at', dir_path + filename_fig + '.pdf')
print('--------------------------------------------------------------------------------')


"""
Example 2.1

Linear setting. Estimate the treatment effect given a *specific* lambda. 
"""
true_te = 0.5 # true treatment effect
eps = 0.05 # bias in observational data
exp_model ='response_func' # 'aipw', 'response_func', 'mean_diff' - see comments in causal_sim.py
d_exp = 5 # dimensions of covariates in experimental data
d_obs = 6 # dimensions of covariates in observational data

true_coef = np.random.normal(size=d_exp+d_obs) # linear model coefficients to generate data
true_coef_exp = true_coef[0:d_exp]
true_coef_obs = true_coef[d_exp:d_exp+d_obs]
Z_exp, A_exp, Y_exp = generate_data(1000, d_exp, true_coef_exp, true_te, true_pi_func, noise=1) 
Z_obs, A_obs, Y_obs = generate_data(2000, d_obs, true_coef_obs, true_te + eps, true_pi_func, noise=1)

X_exp = np.concatenate((Z_exp, A_exp.reshape(-1, 1), Y_exp.reshape(-1, 1)), axis=1) # structured as: covariates, treatment, outcome
X_obs = np.concatenate((Z_obs, A_obs.reshape(-1, 1), Y_obs.reshape(-1, 1)), axis=1)

lambda_ = 0.7 # set a specific lambda
model = model_class(mode='linear', d_exp=d_exp, d_obs=d_obs, exp_model='response_func')
model.fit_model(lambda_, X_exp, X_obs)
estimate_given_lambda = model.beta().item()

print('--------------------------------------------------------------------------------')
print('Linear setting')
print('True treatment effect:', true_te)
print(f"Given lambda_={lambda_}, estimate of treatment effect = {estimate_given_lambda:.3f}", )

"""
Example 2.2

Linear setting. Using cross-validation to select a lambda.
 
"""
lambda_bin = 10 # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
Q_values, lambda_opt, theta_opt = cross_validation(X_exp, X_obs, lambda_vals, mode='linear', k_fold=5, d_exp=d_exp, d_obs=d_obs, exp_model=exp_model, random_state=2024)
estimate = theta_opt.beta().item()
print(f"\nBy cross-validation, our method selects lambda = {lambda_opt:.2f}" )
print(f"Given lambda={lambda_opt:.2f}, our estimate is {estimate:.3f}")

# plot the cross-validation objective function
plt.figure()
plt.title('Cross-validation objective')
plt.plot(lambda_vals, np.log(Q_values), color='black')
plt.xlabel(r'$\lambda$',)
plt.ylabel(r'log CV($\lambda$)')
plt.tight_layout()
filename_fig="example_use_linear"
plt.savefig(dir_path + filename_fig + '.pdf', format='pdf')
plt.close()
print('figure saved at', dir_path + filename_fig + '.pdf')
print('--------------------------------------------------------------------------------')

