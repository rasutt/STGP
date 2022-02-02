# Load Plotly
library(plotly)

# Get data
time_site_df <- 
  read.csv("Hourly avg AQ and weather data 09-04-2021 with site data.csv")

# Get sites and variables
sites <- unique(time_site_df$Site)
vars <- names(time_site_df)[-1]
n_sites <- length(sites)

# Plot site locations
with(time_site_df[1:n_sites, ], {
  plot(Long.East, Lat.South, main = "Site locations", type = 'n', asp = 1)
  text(Long.East, Lat.South, labels = sites, cex = 0.5)
})
plot_ly(time_site_df[1:n_sites, ], x = ~Long.East, y = ~Lat.South, 
        type = 'scatter', mode = 'text', text = ~Site, 
        textfont = list(size = 8)) %>% layout(yaxis = list(scaleanchor = "x"))

# Plot site locations with altitude - Can't get aspect ratio working
plot_ly(time_site_df[1:n_sites, ], x = ~Long.East, y = ~Lat.South, z = ~Alt.AHD, 
        type = 'scatter3d', mode = 'text', text = ~Site, 
        textfont = list(size = 8)) %>%
  layout(scene = list(aspectmode = 'manual', 
                      aspectratio = list(x = 1, y = 1, z = 1/111e3),
                      camera = list(eye = list(x = 0.01, y = -2.2, z = 1))))

# Check proportion data missing for each variable
var_pn_na <- round(sort(colMeans(is.na(time_site_df))), 2)
print(var_pn_na)

# Looks like might wanna skip solar, carbon monoxide, rain, and ammonia data,
# unless data on this day was unusually bad for them.

# Function to get data for all times (rows) and vars (columns) for one site
make_var_mat <- function(site) {
  time_site_df[time_site_df$Site == site, ]
}

# Check proportion data missing for each variable
round(sort(sapply(sites, function(s) mean(is.na(make_var_mat(s))))), 2)

# Might also wanna skip Tamworth, Richmond, and Lindfield.

# Function to get data for all times (rows) and sites (columns) for one variable
make_var_mat <- function(variable) {
  sapply(sites, function(s) variable[time_site_df$Site == s])
}

# Function to plot a variable over the day at all sites
plot_var <- function(variable, name) {
  mat <- make_var_mat(variable)
  matplot(mat, type = 'l', lty = 1, col = 1:n_sites, xlab = 'Time', ylab = name,
          main = paste0(name, ", all sites, 9-4-21"))
  legend("topleft", col = 1:n_sites, lty = 1, legend = sites, cex = 0.5)
}

# Plot variables over the day at all sites
plot_var(time_site_df$TEMP.1h.C, "Temperature")
plot_var(time_site_df$PM10.1h.mcrg.m3, "Particles - 10")
plot_var(time_site_df$NO.1h.pphm, "Nitric Oxide")
plot_var(time_site_df$NO2.1h.pphm, "Nitrous Oxide")
plot_var(time_site_df$WDR.1h, "Wind Direction")
plot_var(time_site_df$RAIN.1h.mm.m2, "Rain")

make_var_mat(time_site_df$RAIN.1h.mm.m2)
make_var_mat(time_site_df$PM10.1h.mcrg.m3)
