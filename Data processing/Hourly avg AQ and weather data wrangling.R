# Read in air quality and meteorology dataset
# list.files()
df <- read.csv("Raw hourly data.csv", skip = 2, 
               header = T)

# Tidy up column-names
df_names <- names(df)
df_names <- gsub(".average.", "", df_names, fixed = T)
df_names <- gsub("rolling.", "rlg.", df_names, fixed = T)
df_names <- gsub("Â", "", df_names, fixed = T)
df_names <- gsub("\\.\\.+", ".", df_names)
df_names <- gsub("\\.$", "", df_names)
df_names <- gsub("µ", "mcr", df_names, fixed = T)
df_names <- gsub("³", "3", df_names, fixed = T)
df_names <- gsub("²", "2", df_names, fixed = T)
names(df) <- df_names

# Function to get data for each site for one variable at one time
make_var_df <- function(var_name, time = 1) {
  if (length(grep(var_name, df_names, fixed = T)) == 0) 
    stop("Wrong variable name:", var_name)
  cols <- grep(var_name, df_names, fixed = T)
  df <- data.frame(gsub(paste0(".", var_name), "", names(df)[cols]), 
                   unlist(df[time, cols]))
  names(df) <- c("Site", var_name)
  df
}

# Function to get data for each site for all variables at one time
make_time_df <- function(var_names, time = 1) {
  # Make list with a data frame for each real variable for one time
  df_list <- lapply(var_names, make_var_df, time = time)
  
  # Combine the data frames to get all the real variables for each site at one
  # time
  new_df <- df_list[[1]]
  for (i in 2:length(df_list)) {
    new_df <- merge(new_df, df_list[[i]], by = "Site", all = T)
  }
  
  # Add time column after site
  new_df <- cbind(Site = new_df[, 1], Time = time, new_df[, -1])
  
  # # Check that all data retained
  # print("All data retained?")
  # print(prod(dim(new_df)) - 2 * nrow(new_df) - sum(is.na(new_df)) ==
  #         prod(dim(df[Time, ])) - sum(is.na(df[Time, ])) - 
  #         2 * nrow(df[Time, ])) - sum(is.na(df[Time, ]))
  
  # Return data frame
  new_df
}

# Define variables
var_names <- c("TEMP.1h.C", "HUMID.1h", "RAIN.1h.mm.m2", "WSP.1h.m.s", "WDR.1h", 
               "SOLAR.1h.W.m2", "PM2.5.1h.mcrg.m3", "PM10.1h.mcrg.m3", 
               "CO.1h.ppm", "CO.8h.rlg.ppm", "SO2.1h.pphm", "NO.1h.pphm", 
               "NO2.1h.pphm", "SD1.1h", "NEPH.1h.bsp", "OZONE.1h.pphm", 
               "OZONE.4h.rlg.pphm", "NH3.1h.pphm")

# Number of times
n_t <- nrow(df)

# Stack data frames over all times
time_df <- make_time_df(var_names, time = 1)
for (t in 2:n_t) {
  time_df <- rbind(time_df, make_time_df(var_names, time = t))
}

# Check that all data retained
print("All data retained?")
print((prod(dim(time_df)) - 2 * nrow(time_df) - sum(is.na(time_df))) ==
        (prod(dim(df)) - 2 * nrow(df) - sum(is.na(df))))

# Read in station information dataset
# list.files()
site_df <- read.csv("Raw sites data.csv", header = T)

# Merge with site data
time_df[, 1] <- gsub(".", " ", time_df[, 1], fixed = T)
site_df[, 2] <- toupper(site_df[, 2])
names(site_df)[c(2, 6:8)] <- c("Site", "Lat.South", "Long.East", "Alt.AHD")
time_site_df <- merge(site_df[, c(2, 6:8)], time_df, by = "Site")
time_site_df <- time_site_df[order(time_site_df[, 5]), ]
row.names(time_site_df) <- NULL

# Show results
print(head(time_site_df[, 1:6], 18))
print(tail(time_site_df[, 1:6], 18))

# Write to csv
write.csv(x = time_site_df, 
          file = "Hourly avg AQ and weather data 09-04-2021 with site data.csv",
          row.names = F)
