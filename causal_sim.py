import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch
import scipy.stats as stats

random_seed = 2024
np.random.seed(random_seed)

def generate_data(n, d, true_mu_coef, true_te, pi_func, noise=0.1, rng=None):
    """ 
    Generate with-covariate data.
    
    Args:
        n: number of data points
        d: dimension of covariates
        true_mu_coef: linear model weights used to generate data
        true_te: true treatment effect
        pi_func: propensity score function
        noise: standard deviation of noise 
        rng: random number generator

    Return:
        X: covariates
        A: binary treatments
        Y: observed outcomes
    """
    if rng == None:
        X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n) 
        pi = pi_func(X)
        A = np.array([np.random.binomial(1, pi[i])  for i in range(n)])
        Y =  X @ true_mu_coef + A * true_te + noise * np.random.randn(n)
    else: 
        X = rng.multivariate_normal(np.zeros(d), np.eye(d), size=n) 
        pi = pi_func(X)
        A = np.array([rng.binomial(1, pi[i])  for i in range(n)])
        Y =  X @ true_mu_coef + A * true_te + noise * rng.normal(size=n)
    return X, A, Y


class model_class(torch.nn.Module):
    """ Class to define models. We use torch here to enable future extensions that require gradient descent"""
    def __init__(self, mode='mean', d_exp=None, d_obs=None, exp_model='aipw', stratified_kfold=False, rng=None):
        """ 
        Args:
            mode: 'mean' - no-covariate setting; 'linear' - linear setting
            d_exp: in linear setting, dimension of covariates in experimental data (exclude treatment)
            d_obs: in linear setting, dimension of covariates in observational data (exclude treatment)
            exp_model: in linear setting, estimator on experimental data used to compute the experimental loss, can be:
                'aipw' - the AIPW estimator
                'mean_diff' - the outcome mean difference between treated
                'response_func' - the plug-in estimator using linear model; where the estimate is just the treatment coefficient
            stratified_kfold: True if stratify by treatment assignment when splitting data in cross-validation, False otherwise
            rng: random number generator
        """
        super(model_class, self).__init__()
        assert mode in ['mean', 'linear'], f"mode must be valid, got: {mode}"
        self.mode = mode
        self.theta = None # theta(...) is to predict with input theta, currently only used in no-covariate setting
        self.theta_model = None # currently only used in linear setting
        self.d_exp = d_exp
        self.d_obs = d_obs
        self.exp_model = exp_model # for linear mode
        self.stratified_kfold = stratified_kfold 
        self.rng = rng
        if self.mode == 'linear':
            assert d_obs is not None, "number of covariates of obs (d_obs) must be specified in the linear setting"
            self.theta_model = torch.nn.Parameter(torch.zeros(d_obs + 2))  # [Treatment coef, other coef, bias]      
                    
    def forward(self):
        if self.mode == 'linear':
            return self.theta_model     
    
    def mean_est(self, lambda_, X_exp, X_obs):
        '''
        The closed-form solution for no-covariate setting.
        
        Args:
            lambda_: given lambda value
            X_exp: experimental data
            X_obs: observational data
        
        Return: estimated quantity given specific lambda
        '''
        return (1 - lambda_) * np.mean(X_exp) + lambda_ * np.mean(X_obs)
        
    def fit_model(self, lambda_, X_exp, X_obs):
        """
        Fit the model given specific lambda using both sources of data.
        
        Args:      
            lambda_: given lambda value
            X_exp: experimental data
            X_obs: observational data
            
        """
        if self.mode == 'mean':
            self.theta = self.mean_est
        if self.mode == 'linear':
            # use close form solution    
            beta_exp_precompute = compute_exp_minmizer(X_exp, mode=self.mode, exp_model=self.exp_model, d_exp=self.d_exp)
            if lambda_ == 0: 
                # directly return minimizer of exp, since otherwise l_matrix is singular in close form solution 
                padded_mini = torch.zeros(self.d_obs + 2)
                padded_mini[0] = beta_exp_precompute
                self.theta_model = torch.nn.Parameter(padded_mini)
                return
            
            Z = X_obs[:, :self.d_obs]
            A = X_obs[:, self.d_obs]
            intercept = torch.ones((X_obs.shape[0], 1))
            Y = X_obs[:, -1] 
            
            A_Z = torch.cat((torch.tensor(A).reshape(-1, 1), torch.tensor(Z), intercept), dim=1) # n_obs * d+1
            e1 = torch.zeros(self.d_obs + 1 + 1) # + intercept
            e1[0] = 1
            e1 = e1.reshape(-1, 1) # d+2 * 1
            l_matrix =  (1 - lambda_ ) * e1 @ e1.T + lambda_ / X_obs.shape[0] * A_Z.T @ A_Z
            r_matrix = (1 - lambda_ ) * beta_exp_precompute *  e1 + lambda_ / X_obs.shape[0] * A_Z.T @ torch.tensor(Y).reshape(-1, 1)
            det = torch.det(l_matrix)
            if torch.isclose(det, torch.tensor(0.0, dtype=det.dtype), atol=1e-7):
                # min norm solution closed-form 
                if lambda_ == 1 and (np.array_equal(A, np.ones_like(A)) or np.array_equal(A, np.zeros_like(A))):
                    # all treated or all untreated (multicollinearity)
                    A_Z = torch.cat((torch.tensor(Z), intercept), dim=1)  # exclude treatment
                    l_matrix = lambda_ / X_obs.shape[0] * A_Z.T @ A_Z
                    r_matrix = lambda_ / X_obs.shape[0] * A_Z.T @ torch.tensor(Y).reshape(-1, 1)
                    solution = l_matrix.T @ (l_matrix @ l_matrix.T).inverse() @ r_matrix
                    solution = torch.cat((torch.tensor([0.0]), solution.reshape(-1)), dim=0) # add 0 as treatment coef
                else: 
                    solution = l_matrix.T @ (l_matrix @ l_matrix.T).inverse() @ r_matrix
                self.theta_model = torch.nn.Parameter(solution.reshape(-1))
            else:
                solution = torch.linalg.solve(l_matrix, r_matrix)
                self.theta_model = torch.nn.Parameter(solution.reshape(-1))      

    
    def beta(self, lambda_=None, X_exp=None, X_obs=None):
        '''
        Obtain beta(theta). This is to extract the treatment effect estimate from the full model theta.

        Args:      
            lambda_: given lambda value
            X_exp: experimental data
            X_obs: observational data
        '''
        if self.mode == 'mean':
            return self.theta(lambda_, X_exp, X_obs)
        if self.mode == 'linear':
            return self.theta_model[0]
     
        
def compute_exp_minmizer(X, mode='linear', exp_model='aipw', stratified_kfold=False, d_exp=None, rng=None): 
    ''' 
    Compute the estimate on experimental data.
    
    Args:
        X: experimental data
        mode: only supports 'linear'
        exp_model: 'aipw' or 'mean_diff' or 'response_func'
            'aipw' - the AIPW estimator, using the true_pi_func as propensity score;
            'mean_diff' - the outcome mean difference between treated;
            'response_func' - the plug-in estimator using linear model; where the estimate is just the treatment coefficient
        stratified_kfold: True to stratify for train/inference splitting on treatment
        d_exp: in linear setting, dimension of covariates in experimental data (exclude treatment)
        rng: random number generator
        
    Return: estimated treatment effect from experimental data
    '''
    if mode == 'mean':
        return 
    if mode == 'linear':
        assert d_exp is not None, "please specify d_exp in compute_exp_minmizer"
        Z_exp_all = X[:, :d_exp]
        A_exp_all = X[:, d_exp]
        Y_exp_all = X[:, -1] 
        if exp_model == 'aipw' and stratified_kfold:
            tv_split = StratifiedKFold(n_splits=2)
            numerator = tv_split.split(Z_exp_all, A_exp_all)
            for train_index, val_index in numerator:
                X_t, X_v = X[train_index], X[val_index]
                Z_t, A_t, Y_t = X_t[:, :d_exp], X_t[:, d_exp], X_t[:, -1] 
                Z_v, A_v, Y_v = X_v[:, :d_exp], X_v[:, d_exp], X_v[:, -1]
                break # one split suffices
        else: 
            # naively split into 2 folds
            n_val =  int(0.5 * X.shape[0]) 
            Z_t, A_t, Y_t = Z_exp_all[n_val:,:], A_exp_all[n_val:], Y_exp_all[n_val:]
            Z_v, A_v, Y_v =  Z_exp_all[:n_val,:], A_exp_all[:n_val], Y_exp_all[:n_val]
     
        if exp_model == 'aipw':
            exp_model = LinearRegression()
            exp_model.fit(np.concatenate((A_t.reshape(-1, 1), Z_t), axis=1), Y_t)
            mu_pred_1 = exp_model.predict(np.concatenate((np.ones((np.shape(Z_v)[0], 1)), Z_v), axis=1))
            mu_pred_0 = exp_model.predict(np.concatenate((np.zeros((np.shape(Z_v)[0], 1)), Z_v), axis=1))
            pi = true_pi_func(Z_v)
            psi = (A_v / pi) * (Y_v - mu_pred_1) + mu_pred_1 - (((1 - A_v) / (1 - pi)) * (Y_v - mu_pred_0) + mu_pred_0)
            beta_exp = np.mean(psi)
        if exp_model == 'mean_diff':
            beta_exp = Y_exp_all[A_exp_all == 1].mean() - Y_exp_all[A_exp_all == 0].mean()
        if exp_model == 'response_func':
            if (np.array_equal(A_exp_all, np.ones_like(A_exp_all)) or np.array_equal(A_exp_all, np.zeros_like(A_exp_all))):
                return 0.0 
            exp_model = LinearRegression()
            exp_model.fit(np.concatenate((A_exp_all.reshape(-1, 1), Z_exp_all), axis=1), Y_exp_all)
            beta_exp = exp_model.coef_[0]
        return beta_exp


def L_exp(beta, X, mode='mean', beta_exp_precompute=None, exp_model='aipw', stratified_kfold=False, d_exp=None, rng=None):
    ''' 
    Compute the loss on experimental data.
    
    Args:
        X: experimental data
        mode: 'mean' - no-covariate setting; 'linear' - linear setting
        exp_model: 'aipw' or 'mean_diff' or 'response_func'
            'aipw' - the AIPW estimator;
            'mean_diff' - the outcome mean difference between treated;
            'response_func' - the plug-in estimator using linear model; where the estimate is just the treatment coefficient
        beta_exp_precompute: precomputed experimental estimate, if applicable, to speed up computation
        stratified_kfold: True to stratify for train/inference splitting on treatment
        d_exp: dimension of covariates in experimental data (exclude treatment)
        
    Return: loss of scalar beta on experimental data
    '''
    if mode == 'mean':
        return np.mean((X - beta) ** 2)  # experiments were run under np.sum, but should be equivalent 
    if mode == 'linear':
        if beta_exp_precompute == None:
            assert d_exp is not None, "please specify d_exp in L_exp"
            beta_exp = compute_exp_minmizer(X, mode=mode, exp_model=exp_model, stratified_kfold=stratified_kfold, d_exp=d_exp, rng=rng)
            beta_exp = torch.tensor(beta_exp)
        else: 
            beta_exp = torch.tensor(beta_exp_precompute)
        return (beta_exp - beta)**2
        

def L_obs(theta_model, X, mode='mean', d_obs=None):
    ''' 
    Compute the loss on observational data.
    
    Args:
        X: observational data
        theta_model: observational model
        mode: 'mean' - no-covariate setting; 'linear' - linear setting
        d: in linear setting, dimension of covariates in observational data (exclude treatment)
        
    Return: loss of observational model on observational data
    '''
    if mode == 'mean':
        return np.mean((np.mean(X) - theta_model) ** 2) # experiement were run under np.sum, but should be equivalent 
    if mode == 'linear':
        assert d_obs is not None, "please specify d_obs in L_obs"
        X = torch.tensor(X, dtype=torch.float64)
        Z = X[:, :d_obs]
        A = X[:, d_obs]
        Y = X[:, -1] 
        X_pred = torch.matmul(torch.cat([A.view(-1, 1), Z], dim=1), theta_model[:-1]) + theta_model[-1] 
        return torch.mean((X_pred - Y) ** 2)
   
        
def combined_loss(theta, X_exp, X_obs, lambda_, mode='mean', beta_exp_precompute=None, exp_model='aipw', stratified_kfold=False, d_exp=None, d_obs=None, rng=None):
    ''' 
    Compute the combined loss on experimental and observational data.
    
    Args:
        theta: obervational model class
        X_exp: experimental data
        X_obs: observational data
        lambda_: lambda value
        mode: 'mean' - no-covariate setting; 'linear' - linear setting
        beta_exp_precompute: precomputed experimental estimate, if applicable, to speed up computation
        exp_model: in linear setting, 'aipw' or 'mean_diff' or 'response_func'
            'aipw' - the AIPW estimator;
            'mean_diff' - the outcome mean difference between treated;
            'response_func' - the plug-in estimator using linear model; where the estimate is just the treatment coefficient
        stratified_kfold: True to stratify for train/inference splitting on treatment
        d_exp: in linear setting, dimension of covariates in experimental data (exclude treatment) 
        d_obs: in linear setting, dimension of covariates in observational data (exclude treatment)
        rng: random number generator
        
    Return: combined loss
    '''
    
    if mode == 'mean':
        combined = (1 - lambda_) * L_exp(theta.beta(lambda_, X_exp, X_obs), X_exp, mode=mode) +  lambda_ * L_obs(theta(lambda_, X_exp, X_obs), X_obs, mode=mode) 
    if mode == 'linear': 
        assert d is not None, "please specify d in combined_loss"
        loss_exp = L_exp(theta.beta(), X_exp, mode=mode, beta_exp_precompute=beta_exp_precompute, exp_model=exp_model, stratified_kfold=stratified_kfold, d_exp=d_exp, rng=rng)
        loss_obs = L_obs(theta.theta_model, X_obs, mode=mode, d_obs=d_obs) 
        combined = (1 - lambda_) * loss_exp + lambda_  * loss_obs
    return combined
    

def cross_validation(X_exp, X_obs, lambda_vals, mode='mean', k_fold=None, d_exp=None, d_obs=None, exp_model='aipw', stratified_kfold=False, random_state=None, rng=None):
    """
    Calculate the cross validation error for each lambda.
    Args: 
        X_exp: experimental data
        X_obs: observational data
        lambda_vals: candidate lambda values
        mode: 'mean' - no-covariate setting; 'linear' - linear setting
        k_fold: number of folds
        d_exp: in linear setting, dimension of covariates in experimental data (exclude treatment)
        d_obs: in linear setting, dimension of covariates in observational data (exclude treatment)
        exp_model: 'aipw' or 'mean_diff' or 'response_func'
            'aipw' - the AIPW estimator;
            'mean_diff' - the outcome mean difference between treated;
            'response_func' - the plug-in estimator using linear model; where the estimate is just the treatment coefficient
        stratified_kfold: True if stratified
        random_state: control shuffle or not across each run
        rng: random number generator

    Return:
        Q_values: a numpy vector for cross validation error of each lambda
        lambda_opt: lambda value that optimizes the cross validation error
        theta_opt: a model_class object, with the fitted model and beta function, though need to put lambda_opt as input 
    """
    if k_fold == None:
        cross_validator = LeaveOneOut()
    else: 
        if stratified_kfold:
            cross_validator = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)
        else: 
            cross_validator = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    Q_values = np.zeros_like(lambda_vals)
    for i, lambda_ in enumerate(lambda_vals):
        current_Q = 0
        if stratified_kfold:
            numerator = cross_validator.split(X_exp, X_exp[:, d_exp])
        else:
            numerator = cross_validator.split(X_exp)
        for train_index, val_index in numerator:
            # fit a model for each fold
            model = model_class(mode=mode, d_exp=d_exp, d_obs=d_obs, exp_model=exp_model, stratified_kfold=stratified_kfold, rng=rng)
            X_train, X_val = X_exp[train_index], X_exp[val_index]
            model.fit_model(lambda_, X_train, X_obs)
            with torch.no_grad():
                l_exp_fold = L_exp(model.beta(lambda_, X_train, X_obs), X_val, mode=mode, exp_model=exp_model, stratified_kfold=stratified_kfold, d_exp=d_exp, rng=rng)
                current_Q += l_exp_fold.item()
        Q_values[i] += current_Q
    Q_values /= X_exp.shape[0]
    lambda_opt = lambda_vals[np.argmin(Q_values)] # optimal lambda
    theta_opt = model_class(mode=mode, d_exp=d_exp, d_obs=d_obs, exp_model=exp_model, stratified_kfold=stratified_kfold, rng=rng)
    theta_opt.fit_model(lambda_opt, X_exp, X_obs) # fitted model on full data using optimal lambda
    return Q_values, lambda_opt, theta_opt


def true_pi_func(X):
    """
    True propensity score function that could be customized.
    
    Args:
        X: covariates

    Return: a vector of propensity score
    """
    return 0.5 * np.ones(X.shape[0])

def tilde_pi_func(X):
    """ 
    Propensity score function that could be customized.

    Args:
        X: covariates

    Return: a vector of propensity score
    """
    return 0.2 * np.ones(X.shape[0]) # return just 0.2
    #return (X[:, 0] > 0).astype(int) # return depending on the first covariate > 0 or not

def lalonde_get_data(df, group, variables, subsample_idx=None):
    '''
    Select the samples given group and the variables in the LaLonde dataset

    Args:
        df: LaLonde dataset
        group: observational control group
        variables: variables for linear model
        subsample_idx: if not None, subsample the same set of experimental data (to construct both experimental and observational data)
    
    Return:
        X_exp: experimental data
        X_obs: observational data
    '''
    X_df = df[df['group'].isin(['control', 'treated'])]
    Z_exp = X_df[variables].to_numpy()
    A_exp = X_df['treatment'].to_numpy()
    Y_exp = X_df['re78'].to_numpy()
    X_exp = np.concatenate((Z_exp, A_exp.reshape(-1, 1), Y_exp.reshape(-1, 1)), axis=1)
    if subsample_idx is not None:
        X_exp = X_exp[subsample_idx]
        X_df = df[df['group'] == group]
    else:
        X_df = df[df['group'].isin(['treated', group])]
    Z_obs = X_df[variables].to_numpy()
    A_obs = X_df['treatment'].to_numpy()
    Y_obs = X_df['re78'].to_numpy()
    X_obs = np.concatenate((Z_obs, A_obs.reshape(-1, 1), Y_obs.reshape(-1, 1)), axis=1)
    if subsample_idx is not None:
        X_df = df[df['group'].isin(['control', 'treated'])].iloc[subsample_idx]
        X_df = X_df[X_df['group'] == 'treated']
        Z_exp = X_df[variables].to_numpy()
        A_exp = X_df['treatment'].to_numpy()
        Y_exp = X_df['re78'].to_numpy()
        X_exp_t =  np.concatenate((Z_exp, A_exp.reshape(-1, 1), Y_exp.reshape(-1, 1)), axis=1)
        X_obs = np.concatenate((X_obs, X_exp_t), axis=0)

    return X_exp, X_obs


def t_test_normal_baseline(x_exp, x_obs, alpha_threshold=0.05, equal_var=True):
    '''
    No-covariate case, t-test baseline. 

    Args:
        x_exp: experimental sample
        x_obs: observational sample
        alpha_threshold: significance threshold    
        equal_var: True if two populations are assumed to have the same variance.
    Return:
        x_exp sample mean if rejecting the null, pooled sample mean otherwise.
    '''
    t_test_result = stats.ttest_ind(x_exp, x_obs, equal_var=equal_var)
    if t_test_result.pvalue < alpha_threshold:
        # reject the null that equal mean
        #print("reject the null that equal mean, use X_exp sample mean")
        return np.mean(x_exp)
    else:
        return np.mean(np.concatenate((x_exp, x_obs), axis=0))