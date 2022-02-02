# Load Plotly and oz packages
library(plotly)
library(oz)

# Get data
date_site_df <- 
  read.csv("Daily avg air quality data 21-04-2020 to 2021 with site data.csv")

# Get sites and variables
sites <- unique(date_site_df$Site)
vars <- names(date_site_df)[-1]
n_sites <- length(sites)

# Plot site locations
# Names
with(date_site_df[1:n_sites, ], {
  plot(Long.East, Lat.South, main = "Site locations", type = 'n', asp = 1)
  text(Long.East, Lat.South, labels = sites, cex = 1)
})
nsw(add = T)

# Names zoomed in
with(date_site_df[1:n_sites, ], {
  plot(Long.East, Lat.South, main = "Site locations", type = 'n', asp = 1,
       xlim = c(150.5, 151.5), ylim = c(-34.5, -33.5))
  text(Long.East, Lat.South, labels = sites, cex = 0.5)
})
nsw(add = T)

# Plotly
plot_ly(date_site_df[1:n_sites, ], x = ~Long.East, y = ~Lat.South, 
        type = 'scatter', mode = 'text', text = ~Site, 
        textfont = list(size = 8)) %>% layout(yaxis = list(scaleanchor = "x"))

# Markers
with(date_site_df[1:n_sites, ], {
  plot(Long.East, Lat.South, asp = 1, main = "Site locations")
})
nsw(add = T)

# Plot site locations with Nitrous Oxide
not_na <- which(!is.na(date_site_df$NO2.24h.pphm[1:n_sites]))
plot_ly(date_site_df[not_na, ], 
        x = ~Long.East, y = ~Lat.South, 
        type = 'scatter', mode = 'markers', color = ~NO2.24h.pphm, 
        alpha = 0.7, text = sites[not_na], hoverinfo = 'text') %>% 
  layout(yaxis = list(scaleanchor = "x"))

# Function to get data for all dates (rows) and vars (columns) for one site
make_site_mat <- function(site) {
  date_site_df[date_site_df$Site == site, ]
}

# Check proportion data missing for each site
site_pn_na <- 
  round(sort(sapply(sites, function(s) mean(is.na(make_site_mat(s))))), 2)
site_pn_na

# Might also wanna skip MORISSET, KATOOMBA, BATHURST, ORANGE, TAMWORTH, ALBURY,
# LINDFIELD, LIVERPOOL SWAQS, MACARTHUR, VINEYARD...

# Check proportion data missing for each variable
var_pn_na <- round(sort(colMeans(is.na(date_site_df))), 2)
var_pn_na

# Looks like might wanna skip sulfur dioxide, carbon monoxide, and PM10 data.
# But a quarter missing even for the best!

# Function to get data for all dates (rows) and sites (columns) for one variable
make_var_mat <- function(variable) {
  sapply(sites, function(s) variable[date_site_df$Site == s])
}

# Function to plot a variable over the day at all sites
plot_var <- function(variable, name, leg = T) {
  mat <- make_var_mat(variable)
  not_na <- colSums(!is.na(mat)) > 0
  matplot(mat[, not_na], type = 'l', lty = 1, col = 1:n_sites, xlab = 'Time', 
          ylab = name, main = paste0(name, ", all sites, 21-04-2020 to 2021"))
  if (leg) {
    legend("topleft", col = 1:n_sites, lty = 1, legend = sites[not_na], 
           cex = 0.4)
  }
}

# Plot variables over the day at all sites
plot_var(date_site_df$NO.24h.pphm, "Nitric Oxide")
plot_var(date_site_df$NO2.24h.pphm, "Nitrous Oxide", leg = F)
plot_var(date_site_df$OZONE.24h.pphm.1, "Ozone")
plot_var(date_site_df$SO2.24h.pphm, "Sulfur Dioxide")
plot_var(date_site_df$CO.24h.ppm, "Carbon Monoxide")
plot_var(date_site_df$PM10.24h.mcrg.m3, "Particles - 10")

# Most of the gases have some small negative values, I guess the instruments
# aren't perfect. From the site, "Full data validation completed up to 30 June
# 2020. Later records have passed initial, automated validation process for
# online display."

# Check correlation of NO and NO2, NO2 has higher values so looks more
# interesting
with(date_site_df, cor(NO.24h.pphm, NO2.24h.pphm, use = "p"))
