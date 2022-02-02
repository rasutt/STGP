# Import external packages
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Import Gaussian process superclass
from .GP import GP

# Spatial Gaussian Process class
class SGP(GP):
    # Constructor method
    def __init__(self, df, target_var, target_name, long_low, long_up, lat_low, 
                 lat_up, date, cov_func_name):
        self.long_low, self.long_up = long_low, long_up
        self.lat_low, self.lat_up = lat_low, lat_up
        self.date = date
        self.np_filter = self.filter(df, target_var)
        self.sites = df['Site'].values[self.np_filter, np.newaxis]
        GP.__init__(self, df, target_var, target_name, cov_func_name)

        # Find max and min latitude and longitude of data
        self.min_long, self.max_long = np.min(self.X[:, 0]), np.max(self.X[:, 0])
        self.min_lat, self.max_lat = np.min(self.X[:, 1]), np.max(self.X[:, 1])

        # Save coordinates of Newcastle and Wollongong to plot the coastline
        # as a straight line between them
        self.X_COAST = np.array([[150.88733, -34.41706], [151.75963, -32.93118], 
                                 [152.8765, -31.44776], [153.1181, -30.29831]])

    def filter(self, df, target_var):
        np_filter = (df['Long.East'] > self.long_low) & (df['Long.East'] < self.long_up) & \
            (df['Lat.South'] > self.lat_low) & (df['Lat.South'] < self.lat_up) & \
            (df['Date'] == self.date) & ~np.isnan(df[target_var])
        return np_filter

    # Function to select data within spatial bounds
    def select(self, df):
        df_filt = df.loc[self.np_filter, :]
        return df_filt[['Long.East', 'Lat.South']].values, \
            df_filt[self.target_var].values[:, np.newaxis]

    # Method to initialise hyperparameters (spatial root variance and lengthscale)
    def init_hps(self):
        # Initial function standard deviation
        hps = [np.sqrt(0.9 * np.var(self.y_std))]
        
        # One lengthscale for squared exponential covariance
        if self.cov_func_name == "Squared exponential":
            hps = hps + [np.prod(np.ptp(self.X_std, axis = 0)) / len(self.y)]
            
        # Two for ARD covariance
        if self.cov_func_name == "ARD":
            l = np.ptp(self.X_std, axis = 0) / np.sqrt(len(self.y))
            hps = hps + [l[0]] + [l[1]]
            
        return hps

    # Method to find covariance given covariate matrices and hyperparameters
    def cov_func(self, x_1, x_2):
        # Squared exponential covariance
        if self.cov_func_name == "Squared exponential":
            K = self.se_cov_func(x_1, x_2, self.hyper_params[0:2])

        # Automatic relevance determination - Squared exponential with seperate 
        # length scale parameters - TBC
        if self.cov_func_name == "ARD":
            K = self.ard_cov_func(x_1, x_2, self.hyper_params[0:3])

        # Return covariance
        return K

    # Function to make mesh
    def enmesh(self, n_mesh):
        # Create vectors/matrices of longitudes and latitudes for 
        # predicting/plotting, with highest latitudes at top
        x_star_long = np.linspace(self.min_long, self.max_long, n_mesh)
        x_star_lat = np.linspace(self.min_lat, self.max_lat, n_mesh)
        x_star_long_mesh, x_star_lat_mesh = np.meshgrid(x_star_long, x_star_lat)

        # Combine coordinates of mesh for predictions/plotting
        x_star = np.c_[x_star_long_mesh.reshape(n_mesh**2), 
                       x_star_lat_mesh.reshape(n_mesh**2)]


        # Return mesh, longitude and latitude values
        return x_star_long, x_star_lat, x_star

    # Function to build mesh over range of data and plot predictions for it
    def plot_preds(self):
        # Print confidence interval coverage
        [f_mean, f_var] = self.predict(self.X)
        f_low, f_up = self.ci(f_mean, f_var, data = True)
        self.print_ci_cov(f_low, f_up, "confidence")
        
        # Make mesh and predict over it
        n_mesh = 30
        x_star_long, x_star_lat, x_star = self.enmesh(n_mesh)
        [f_mean, f_var] = self.predict(x_star)

        # Reshape predictions
        f_mean_mat = f_mean.reshape(n_mesh, n_mesh)
        f_std_mat = self.sd(f_var).reshape(n_mesh, n_mesh)
        f_low_mat = f_mean_mat - 1.96 * f_std_mat
        f_up_mat = f_mean_mat + 1.96 * f_std_mat

        # Plot it all
        self.plot_spatial(n_mesh, x_star_long, x_star_lat, x_star, f_mean_mat, 
                          f_low_mat, f_up_mat, self.X, self.y, self.date)

    # Function to plot posterior credible interval - this is currently for the 
    # data but should make for mean as well 
    def plot_post(self):
        # Sample conditional posterior predictive at data points
        self.samp_cond_pred(self.X, len(self.y), data = True)

        # Print posterior credible interval coverage
        self.print_ci_cov(np.quantile(self.samp_mat, 0.025, axis=0), 
                          np.quantile(self.samp_mat, 0.9725, axis=0),
                          "credible")
        
        # Make mesh and sample over it
        n_mesh = 15
        x_star_long, x_star_lat, x_star = self.enmesh(n_mesh)
        self.samp_cond_pred(x_star, n_mesh**2)
        
        # Reshape predictions
        f_mean_mat = np.mean(self.samp_mat, axis=0).reshape(n_mesh, n_mesh)
        f_low_mat = np.quantile(self.samp_mat, 0.025, axis=0).reshape(n_mesh, n_mesh)
        f_up_mat = np.quantile(self.samp_mat, 0.9725, axis=0).reshape(n_mesh, n_mesh)

        # Plot with mean and 95% credible interval
        self.plot_spatial(n_mesh, x_star_long, x_star_lat, x_star, f_mean_mat, 
                          f_low_mat, f_up_mat, self.X, self.y, self.date)
    
    # Function to plot
    def plot_spatial(self, n_mesh, x_star_long, x_star_lat, x_star, f_mean_mat, 
                     f_low_mat, f_up_mat, X, y, date):
        # Create rotating perspective frames for animation
        n_angles = 16
        t_vec = 3 * np.pi / 2 + (2 * np.pi / n_angles * np.arange(0, n_angles))
        frames = [ dict(layout = dict(scene_camera=dict(
            eye=dict(x = 2.1 * np.cos(t), y = 2.1 * np.sin(t), z = 0.8)))) 
            for t in t_vec ]

        # Play button for animation
        play_but = dict(label="Play", method="animate", args=[None])

        # Plotly animated interactive 3D scatter plot
        fig = self.make_fig(
            X, y, x_star_long, x_star_lat, f_mean_mat, f_low_mat, f_up_mat, x_star, n_mesh, 
            date, self.sites, play_but, frames = frames)

        # Show figure
        fig.show()

    # Make figure for plot
    def make_fig(self, X, y, x_star_long, x_star_lat, f_mean_mat, f_low_mat, f_up_mat,
                x_star, n_mesh, date, sites, play_but, frames = []):
        # Create uniform color matrix for GP surfaces
        uni_col_mat = np.ones(f_mean_mat.shape)

        # Make figure
        fig = go.Figure(
          data = self.data(X, y, x_star_long, x_star_lat, f_mean_mat, f_low_mat, f_up_mat,
                           uni_col_mat, sites) + 
                        self.map(x_star, n_mesh, X, y, x_star_long, x_star_lat), 
          layout = self.layout(date, x_star_lat, play_but),
          frames = frames)
        fig.update_traces(hoverinfo='skip')
        fig.update_traces(showscale=False, selector=dict(type="surface"))

        # Return figure
        return fig

    # Function to make plotly graph object data for map at bottom of plot
    def map(self, x_star, n_mesh, X, y, x_star_long, x_star_lat):
        # Create matrix with colours for map
        neg_grad = (self.X_COAST[1, 1] - self.X_COAST[0, 1]) / \
        (self.X_COAST[1, 0] - self.X_COAST[0, 0])
        map_col_mat = ((x_star[:, 1] - 0.1 - 
                        (x_star[:, 0] - 0.1 - self.X_COAST[1, 0]) * 
                        neg_grad) < self.X_COAST[1, 1]).reshape(n_mesh, n_mesh)

        # Create plotly graph object data for map
        data = [
          # Locations on map at bottom
          go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y[:, 0] * 0,  
                       mode='markers', name="Location", 
                       marker=dict(size=4, color='black', opacity=0.2)),

          # Coastline on map at bottom - Need to adjust for different domains
          go.Scatter3d(x=self.X_COAST[:, 0] + 0.05, y=self.X_COAST[:, 1] - 0.05, 
                        z=np.zeros(4), mode = "lines",
                        line=dict(color='black', width=2), name="Coastline"),

          # Ocean for map at bottom
          go.Surface(x=x_star_long, y=x_star_lat, z = map_col_mat * 1 - 1, 
                      opacity=0.5, surfacecolor = map_col_mat, 
                      colorscale = 'Blues')]

        # Return graph object data
        return data
  
    # Function to make observation and GP prediction graph object data for plot
    def data(self, X, y, x_star_long, x_star_lat, f_mean_mat, f_low_mat, f_up_mat,
           uni_col_mat, sites):
        data=[
              # Target variable levels and sitenames
              go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y[:, 0], mode='markers+text', 
                           marker=dict(size=4, color='red', opacity=0.5),
                           text=sites, textfont=dict(size=6), 
                           name=f"Site {self.target_name} level"),

              # Gaussian process mean surface
              go.Surface(x=x_star_long, y=x_star_lat, z = f_mean_mat, 
                         opacity=0.3, surfacecolor = uni_col_mat),

              # Gaussian process confidence interval surfaces
              go.Surface(x=x_star_long, y=x_star_lat, z = f_low_mat, 
                         opacity=0.2, surfacecolor = uni_col_mat,
                         colorscale = 'Greys'),
              go.Surface(x=x_star_long, y=x_star_lat, z = f_up_mat, 
                         opacity=0.2, surfacecolor = uni_col_mat,
                         colorscale = 'Greys')]

        # Return graph object data
        return data
    
    # Function to make plot layout
    def layout(self, date, x_star_lat, play_but):
        # Pause button for animation
        pause_but = dict(label="Pause", method="animate",
                          args=[None, {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate", 
                                      "transition": {"duration": 0}}]) 

        # Plot layout
        layout = go.Layout(
          width=900, height=700, margin = dict(l=10, r=10, t=40, b=10), 
          title=f"Gaussian Process for {self.target_name} in NSW on date {date}",
          scene=dict(
              xaxis = dict(showspikes=False, range=[self.min_long, self.max_long]),
              yaxis = dict(showspikes=False, range=[self.min_lat, self.max_lat]), 
              zaxis = dict(showspikes=False, range=[0, np.max(self.y)])),
          scene_camera=dict(eye=dict(x=0, y=-2.1, z=0.8)),
          scene_aspectmode='manual', scene_aspectratio=dict(
              x=np.cos(-np.pi / 180 * np.mean(x_star_lat)), y=1, z=0.7), 
          updatemenus=[dict(type="buttons", buttons=[play_but, pause_but])])

        # Return plot layout
        return layout