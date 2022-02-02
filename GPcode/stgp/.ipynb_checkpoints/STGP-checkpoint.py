# Import external packages
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Import Spatial Gaussian Process superclass
from .SGP import SGP

# Spatial Temporal Gaussian Process class
class STGP(SGP):
    # Constructor method
    def __init__(self, df, target_var, target_name, long_low, long_up, lat_low, 
                 lat_up, date_low, cov_func_name):
        self.date_low = date_low
        SGP.__init__(self, df, target_var, target_name, long_low, long_up, lat_low, 
                     lat_up, None, cov_func_name)
        self.min_date = int(np.min(self.X[:, 2]))
        self.max_date = int(np.max(self.X[:, 2]))
        self.dates = np.arange(self.min_date, self.max_date + 1)
        self.n_dates = len(self.dates)
    
    def filter(self, df, target_var):
        np_filter = (df['Long.East'] > self.long_low) & (df['Long.East'] < self.long_up) & \
            (df['Lat.South'] > self.lat_low) & (df['Lat.South'] < self.lat_up) & \
            (df['Date'] > self.date_low) & ~np.isnan(df[target_var])
        return np_filter

    # Function to select data within spatial bounds
    def select(self, df):
        df_filt = df.loc[self.np_filter, :]
        return df_filt[['Long.East', 'Lat.South', 'Date']].values, \
            df_filt[self.target_var].values[:, np.newaxis]

    # Method to initialise hyperparameters (spatial root variance and lengthscale)
    def init_hps(self):
        # Splitting data variance between observation noise and spatial and 
        # temporal signal function, and dividing covariate spaces by number of points for 
        # lengthscale.  Depending on the scaling last implemented the variance of the scaled
        # observations may or may not be equal to one.

        # Function standard deviation
        hps = [np.sqrt(0.9 * np.var(self.y_std))]
        
        # One spatial lengthscale for squared exponential covariance
        if self.cov_func_name == "Squared exponential":
            hps = hps + [np.prod(np.ptp(self.X_std, axis=0)) / np.sqrt(len(self.y))]
            
        # Two spatial lengthscales for ARD covariance
        if self.cov_func_name == "ARD":
            l = np.ptp(self.X_std, axis=0) / np.sqrt(len(self.y))
            hps = hps + [l[0]] + [l[1]]
        
        # Temporal lengthscale
        hps = hps + [np.ptp(self.X_std[:, 2]) / np.sqrt(len(self.y))]
            
        return hps

    # Method to find covariance given covariate matrices and hyperparameters
    def cov_func(self, x_1, x_2):
        # Separable squared exponential spatial temporal covariance
        if self.cov_func_name == "Squared exponential":
            K_s = self.se_cov_func(x_1[:, 0:2], x_2[:, 0:2], self.hyper_params[0:2])
            K_t = self.se_cov_func(x_1[:, 2][:, np.newaxis], x_2[:, 2][:, np.newaxis], 
                                   [1, self.hyper_params[2]])
            K = np.multiply(K_s, K_t)

        # Automatic relevance determination - Squared exponential with seperate 
        # length scale parameters
        if self.cov_func_name == "ARD":
            K_s = SGP.cov_func(self, x_1[:, 0:2], x_2[:, 0:3])
            K_t = self.se_cov_func(x_1[:, 2][:,np.newaxis], x_2[:, 2][:,np.newaxis], 
                                   [1, self.hyper_params[3]])
            K = np.multiply(K_s, K_t)

        # Return covariance
        return K

    # Function to make mesh
    def enmesh(self, n_mesh):
        # Get spatial mesh
        x_star_long, x_star_lat, x_star_spatial = SGP.enmesh(self, n_mesh)

        # Create temporal coordinates of mesh 
        x_star_date_mesh = np.repeat(self.dates[:, np.newaxis], n_mesh**2, axis = 0)
        x_star_spatial_mesh = np.tile(
            x_star_spatial, (int(self.max_date - self.min_date + 1), 1))
        x_star = np.c_[x_star_spatial_mesh, x_star_date_mesh]

        # Return mesh variables
        return x_star_long, x_star_lat, x_star
    
    # Function to plot posterior credible interval - this is currently for the 
    # data but should make for mean as well 
    def plot_post(self, n_mesh = 10):
        # Sample conditional posterior predictive at data points
        self.samp_cond_pred(self.X, len(self.y), data = True)

        # Print posterior credible interval coverage
        self.print_ci_cov(np.quantile(self.samp_mat, 0.025, axis=0), 
                          np.quantile(self.samp_mat, 0.9725, axis=0),
                          "credible")
        
        # Make mesh and sample over it
        x_star_long, x_star_lat, x_star = self.enmesh(n_mesh)
        self.samp_cond_pred(x_star, self.n_dates * n_mesh**2)
        
        # Reshape predictions
        f_mean_ary = np.mean(self.samp_mat, axis=0).reshape(self.n_dates, n_mesh, n_mesh)
        f_low_ary = np.quantile(self.samp_mat, 0.025, axis=0).reshape(self.n_dates, n_mesh, n_mesh)
        f_up_ary = np.quantile(self.samp_mat, 0.9725, axis=0).reshape(self.n_dates, n_mesh, n_mesh)

        # Build plot
        self.plot(n_mesh, x_star_long, x_star_lat, x_star, f_mean_ary, f_low_ary, f_up_ary)

    # Function to build mesh over range of data and plot predictions for it
    def plot_preds(self, n_mesh = 10, save = False):
        # Print confidence interval coverage
        [f_mean, f_var] = self.predict(self.X)
        f_low, f_up = self.ci(f_mean, f_var, data = True)
        self.print_ci_cov(f_low, f_up, "confidence")

        # Make mesh and predict over it
        x_star_long, x_star_lat, x_star = self.enmesh(n_mesh)

        # Predict over mesh
        [f_mean, f_var] = self.predict(x_star)

        # Reshape predictions
        f_mean_ary = f_mean.reshape(self.n_dates, n_mesh, n_mesh)
        f_std_ary = self.sd(f_var).reshape(self.n_dates, n_mesh, n_mesh)
        f_low_ary = f_mean_ary - 1.96 * f_std_ary
        f_up_ary = f_mean_ary + 1.96 * f_std_ary
        
        # Build plot
        self.plot(n_mesh, x_star_long, x_star_lat, x_star, f_mean_ary, f_low_ary, f_up_ary, save)

    # Function build plot
    def plot(self, n_mesh, x_star_long, x_star_lat, x_star, f_mean_ary, f_low_ary, f_up_ary, save):
        # Create uniform color matrix for GP surfaces
        uni_col_mat = np.ones((n_mesh, n_mesh))

        # Make frames for animation
        frames = [None] * (self.n_dates - 1)
        for d in np.arange(1, self.n_dates):
            date = self.dates[d]
            date_idx = (self.X[:, 2] == date)
            frames[d - 1] = go.Frame(
            data = self.data(
                self.X[date_idx, 0:2], self.y[date_idx], x_star_long, x_star_lat,
                f_mean_ary[d, :, :], f_low_ary[d, :, :], f_up_ary[d, :, :], uni_col_mat,
                self.sites[date_idx]),
            layout = go.Layout(dict(
                title=f"Gaussian Process for {self.target_name} in NSW on date {date}")))

        # Buttons for animation
        play_but = dict(label="Play", method="animate", 
                        args=[None, {
                            # "frame": {"duration": 500, "redraw": False},
                            # "fromcurrent": True, 
                            "transition": {"duration": 300,
                                           "easing": "quadratic-in-out"}}])

        # Get first date index
        date_1_idx = self.X[:, 2] == self.dates[0]

        # Make animated interactive 3D scatter plot
        fig = self.make_fig(
            self.X[date_1_idx, 0:2], self.y[date_1_idx], x_star_long, x_star_lat, 
            f_mean_ary[0, :, :], f_low_ary[0, :, :], f_up_ary[0, :, :], x_star[:(n_mesh**2), :],
            n_mesh, self.dates[0], self.sites[date_1_idx], play_but, frames)
        
        # If requested to save plot
        if (save):
            fig.write_image("spatial_temporal_plot.pdf")

        # Show plot
        fig.show()