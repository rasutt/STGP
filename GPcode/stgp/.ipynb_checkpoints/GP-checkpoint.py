# Import external packages
import numpy as np
from scipy.optimize import minimize

# Gaussian Process superclass
class GP:
    # Constructor method
    def __init__(self, df, target_var, target_name, cov_func_name):
        # Store parameters
        self.target_var = target_var
        self.target_name = target_name
        self.X, self.y = self.select(df) # Implemented in subclasses
        self.X_std, self.X_mean, self.X_scale = self.cen_scale(self.X)
        self.y_std, self.y_mean, self.y_scale = self.cen_scale(self.y)
        self.cov_func_name = cov_func_name
        self.hyper_params = self.init_hps() # Implemented in subclasses
        self.sigma_n = np.sqrt(0.1 * np.var(self.y_std)) # Observation noise root variance
        self.K = self.cov_func(self.X_std, self.X_std) # Data covariance
        self.K_inv = np.linalg.pinv(self.K + np.eye(self.K.shape[0]) * 
                                    self.sigma_n**2)
      
    # Function to center and scale data
    def cen_scale(self, X):
        X_mean = np.mean(X, axis = 0)
        X_std = X - X_mean
        X_scale = np.std(X_std, axis = 0)
        X_std = X_std / X_scale
        return X_std, X_mean, X_scale
    
    # Method to find squared distances between covariate matrices 
    def sq_dist(self, x_1, x_2):
        # Sums of squared observations (rows)
        x_1_sq = np.sum(np.square(x_1), 1)
        x_2_sq = np.sum(np.square(x_2), 1)

        # Squared distances (differences) between each observation
        return -2. * np.dot(x_1, x_2.T) + x_1_sq[:, None] + x_2_sq[None, :]

    # Method to find squared exponential covariance given covariate matrices and 
    # hyperparameters
    def se_cov_func(self, x_1, x_2, hyper_params):
        # Squared distances between observations
        d = self.sq_dist(x_1, x_2)

        # Return squared exponential covariance
        return hyper_params[0]**2 * np.exp(-0.5 * d / hyper_params[1]**2)

    # Method to find automatic relevance determination covariance given covariate
    # matrices and hyperparameters
    def ard_cov_func(self, x_1, x_2, hyper_params):
        # Squared distances between observations
        d_1 = self.sq_dist(x_1[:, 0][:, np.newaxis], x_2[:, 0][:, np.newaxis])
        d_2 = self.sq_dist(x_1[:, 1][:, np.newaxis], x_2[:, 1][:, np.newaxis])

        # Return automatic relevance determination covariance
        return hyper_params[0]**2 * \
            np.exp(-0.5 * (d_1 / hyper_params[1]**2 + d_2 / hyper_params[2]**2))

    # Prediction method
    def predict(self, x_star):
        x_star_std = (x_star - self.X_mean) / self.X_scale
        k_star = self.cov_func(self.X_std, x_star_std)
        f_mean = np.transpose(k_star) @ self.K_inv @ self.y_std
        f_var = self.cov_func(x_star_std, x_star_std) - \
            np.transpose(k_star) @ self.K_inv @ k_star
        return [f_mean * self.y_scale + self.y_mean, f_var * self.y_scale**2]

    # Function to load hyperparameter samples
    def load_hp_samps(self, hp_samps):
        self.hp_samps = hp_samps
      
    # Function to set Gaussian process to use posterior means of hyperparameters
    def set_post_mean(self):
        post_mean_thps = np.mean(self.hp_samps, axis = 0)
        print('Posterior means of hyperparameters:')
        print(post_mean_thps)
        self.set_hyper_params(post_mean_thps[:-1], 
                              post_mean_thps[-1])
        
    # Function to set Gaussian process to use maximum a posteriori hyperparameter values
    def set_max_post(self, n_bins=20):
        # Plot posterior distributions of hyperparameters
        self.hp_samps.hist(bins = n_bins);
        
        # Find map estimates
        n_pars = self.hp_samps.shape[1]
        maphps = np.zeros(n_pars)
        for p in range(n_pars):
            counts, bins = np.histogram(self.hp_samps.iloc[:, p], bins = n_bins)
            maphps[p] = ((bins[1:] + bins[:n_bins]) / 2)[np.argmax(counts == np.max(counts))]
        print(maphps)
        self.set_hyper_params(maphps[:-1], maphps[-1])
        print('Maximum a posteriori estimates of hyperparameters:')
        print(self.hp_samps.columns.values)
        print(maphps)
      
    # Function to sample from conditional predictive distribution
    def samp_cond_pred(self, x_star, n_star, n_samps = 500, data = False):
        # Save hyperparameters
        hps = self.hyper_params
        sigma_n = self.sigma_n

        # Create random number generator
        rng = np.random.default_rng()
        
        # Sample from GP for each set of hyperparameter posterior samples
        self.samp_mat = np.ndarray((n_samps, n_star))
        
        print("Sampling posterior predictive distribution")

        # Loop over posterior hyperparameter samples
        for s in range(n_samps):
            if (s % 100 == 0):
                print(f'sample {s + 1}')
                
            # Get conditional posterior predictive mean and variance
            self.set_hyper_params(self.hp_samps.iloc[-s][:-1], 
                                  self.hp_samps.iloc[-s][-1])
            [f_mean, f_var] = self.predict(x_star)

            # Sample from posterior
            self.samp_mat[s, :] = \
                rng.multivariate_normal(mean = f_mean.ravel(), cov = f_var)
            
            # If sampling data add noise term
            if (data):
                self.samp_mat[s, :] = self.samp_mat[s, :] + \
                    rng.standard_normal(n_star) * self.sigma_n
            
        # Restore hyperparameters
        self.set_hyper_params(hps, sigma_n)
  
    # Function to get marginal standard deviations from covariance matrix
    def sd(self, f_var, data = False):
        # If for data include noise variance
        if (data):
            return np.sqrt(np.diagonal(f_var) + self.sigma_n)
        else:
            return np.sqrt(np.diagonal(f_var))
    
    # Function to find 95% confidence interval
    def ci(self, f_mean, f_var, data = False):
        f_mean = f_mean.ravel()
        f_std = self.sd(f_var, data).ravel()
        f_low = f_mean - 1.96 * f_std
        f_up = f_mean + 1.96 * f_std
        
        return(f_low, f_up)

    # Function to print confidence interval coverage
    def print_ci_cov(self, f_low, f_up, interval):
        covered = (self.y > f_low) & (self.y < f_up)
        percent = np.round(np.mean(covered) * 100)
        print(f'{percent}% of data within 95% {interval} interval')

    # Method to change training data and update data covariance
    def set_training_data(self, X, y):
        self.X = X
        self.y = y
        self.X_std, self.X_mean, self.X_scale = self.cen_scale(X)
        self.y_std, self.y_mean, self.y_scale = self.cen_scale(y)
        self.K = self.cov_func(self.X_std, self.X_std)
        self.K_inv = np.linalg.pinv(self.K + np.eye(self.K.shape[0]) * 
                                    self.sigma_n**2)

    # Method to change hyper-parameters and update data covariance
    def set_hyper_params(self, hyper_params, sigma_n):
        self.hyper_params = hyper_params
        self.K = self.cov_func(self.X_std, self.X_std)
        self.sigma_n = sigma_n
        self.K_inv = np.linalg.pinv(
            self.K + np.eye(self.K.shape[0]) * self.sigma_n**2)
      
    # Method to find log marginal likelihood
    def log_marginal_likelihood(self):
        lml1 = -.5 * (np.transpose(self.y_std) @ self.K_inv @ self.y_std)
        lml2 = -.5 * np.log(np.linalg.det(
            self.K + np.eye(self.K.shape[0]) * self.sigma_n**2))
        lml3 = -.5 * self.X_std.shape[0] * np.log(2 * np.pi)
        return (lml1 + lml2 + lml3).ravel()

    # Method to fit model given starting values
    def fit(self, x0 = None, bounds = None, verbose = False):
        print("GP Optimising Hyper-parameters")

        # Sub-method to evaluate negative log marginal likelihood given a 
        # Gaussian Process object and some parameters
        def eval_params(params, gp):
            # Adapted to avoid error when hyperparameters approach zero
            params = np.exp(params) + 1e-3 

            # Set hyper-parameters - last parameter is observation noise root variance
            gp.set_hyper_params(params[:-1], params[len(params) - 1])

            # Find negative log marginal likelihood
            nlml = -gp.log_marginal_likelihood()

            # If verbose print progress
            if verbose:
              print(str(params) + str(nlml))
            return nlml

        # If no starting values provided use current values    
        if x0 is None:
            x0 = self.hyper_params + [self.sigma_n]

        # Display initial hyperparameter values
        print("Initial hyperparameter values: ")
        print(np.round(x0, 3))

        if bounds is not None:
            print('Hyperparameter bounds:')
            print(np.round(np.exp(bounds), 3).T)

        # Minimise negative log marginal likelihood, passing in starting values 
        # and Gaussian Process object
        res = minimize(eval_params, x0, method = 'L-BFGS-B', args = (self), 
                       bounds = bounds)

        if res.success:
            print("Optimisation results: ")
            # Adapted to avoid error when hyperparameters approach zero
            optimal_params = np.exp(res.x) + 1e-3 
            print(np.round(optimal_params, 3))
            print("Log marginal likelihood")
            print(self.log_marginal_likelihood())
            self.set_hyper_params(optimal_params[:-1], 
                                optimal_params[len(optimal_params) - 1])
        else:
            print("Error when fitting GP.")