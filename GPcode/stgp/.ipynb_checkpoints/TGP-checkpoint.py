# Import external packages
import numpy as np
import plotly.graph_objects as go

# Import Gaussian process superclass
from .GP import GP

# Temporal Gaussian Process class
class TGP(GP):      
    # Constructor method
    def __init__(self, df, target_var, target_name, site, date_low,  
                 cov_func_name):
        self.site = site
        self.date_low = date_low
        GP.__init__(self, df, target_var, target_name, cov_func_name)
    
    # Function to select data after given date for given site
    def select(self, df):
        np_filter = (df['Site'] == self.site) & (df['Date'] > self.date_low) & \
            ~np.isnan(df[self.target_var])
        df_filt = df.loc[np_filter, :]

        return df_filt['Date'].values[:, np.newaxis], \
            df_filt[self.target_var].values[:, np.newaxis]

    # Method to initialise hyperparameters (temporal root variance and lengthscale)
    def init_hps(self):
        return [np.sqrt(0.9 * np.var(self.y_std)), np.ptp(self.X_std) / len(self.y)]

    # Method to find covariance given covariate matrices and hyperparameters
    def cov_func(self, x_1, x_2):
        # Squared exponential covariance
        if self.cov_func_name == "Squared exponential":
            K = self.se_cov_func(x_1, x_2, self.hyper_params[0:2])

        # Sinusoidal covariance - Not finished
        if self.cov_func_name == "Sinusoidal":
            # Squared temporal distances (differences) between each 
            # observation - newaxis stuff necessary?
            d = self.sq_dist(x_1[:, np.newaxis], x_2[:, np.newaxis])

            # Period of model function - this'll have to be adjusted
            P = 1

            # Sinusoidal temporal covariance - varies between max and min 
            # depending on distance mod period
            K = self.hyper_params[2]**2 * \
              np.exp(-2 * np.sin(np.pi * (np.sqrt(d) / P))**2 / 
                     self.hyper_params[3]**2)

        # Return covariance
        return K
  
    # Create mesh for predictions
    def enmesh(self, n_mesh = 100):
        return np.linspace(np.min(self.X), np.max(self.X), 100)[:, np.newaxis]

    # Function to plot predictions
    def plot_preds(self):
        # Print confidence interval coverage
        [f_mean, f_var] = self.predict(self.X)
        f_low, f_up = self.ci(f_mean, f_var, data = True)
        self.print_ci_cov(f_low, f_up, "confidence")
        
        # Create mesh for predictions
        x_star = self.enmesh()

        # Predict from fit model
        [f_mean, f_var] = self.predict(x_star)
        x_star = x_star.flatten()
        f_mean = f_mean.flatten()
        f_std = self.sd(f_var).flatten()
        f_low = f_mean - 2 * f_std
        f_up = f_mean + 2 * f_std
        
        # Plot with mean and 95% confidence interval
        self.plot(x_star, f_mean, f_low, f_up)
        
    # Function to plot posterior credible interval - this is currently for the 
    # data but should make for mean as well 
    def plot_post(self):
        # Sample conditional posterior predictive at data points
        self.samp_cond_pred(self.X, len(self.y), data = True)

        # Print posterior credible interval coverage
        self.print_ci_cov(np.quantile(self.samp_mat, 0.025, axis=0), 
                          np.quantile(self.samp_mat, 0.9725, axis=0),
                          "credible")
        
        # Create mesh for predictions
        n_mesh = 100
        x_star = self.enmesh(n_mesh)

        # Sample conditional posterior predictive over mesh
        self.samp_cond_pred(x_star, n_mesh)
        
        # Plot with mean and 95% credible interval
        self.plot(x_star.flatten(), np.mean(self.samp_mat, axis=0),
                  np.quantile(self.samp_mat, 0.025, axis=0), 
                  np.quantile(self.samp_mat, 0.9725, axis=0))
    
    # Function to plot with mean and 95% confidence interval
    def plot(self, x_star, f_mean, f_low, f_up):
        # Plot predictions over time
        fig = go.Figure(data=[
            go.Scatter(x=x_star, y=f_low.flatten(), fill=None, mode='lines', 
                       line_color='grey', name="Lower confidence bound"),
            go.Scatter(x=x_star, y=f_up.flatten(), fill='tonexty', mode='lines', 
                       line_color='grey', name="Upper confidence bound"),
            go.Scatter(x=x_star, y=f_mean.flatten(), mode='lines', 
                       line_color='red', opacity=0.5,
                       name="Gaussian process predictions"),
            go.Scatter(x=self.X.ravel(), y=self.y.ravel(), mode='markers', 
                       marker=dict(color='red', opacity=0.5),
                       name=f"{self.site} {self.target_name} level")],
            layout=dict(title=f'Daily Average {self.target_name} Concentration at {self.site}',
                        xaxis_title="Date", yaxis_title=self.target_name))
        fig.update_traces(hoverinfo='skip')
        fig.show()