# Read in air quality only dataset (daily averages)
# list.files()
df <- read.csv("Raw daily data.csv", skip = 2, 
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

# Function to get data for each site for one variable at one date
make_var_df <- function(var_name, date = 1) {
  # Stop if variable doesn't exist
  if (length(grep(var_name, df_names, fixed = T)) == 0) 
    stop("Wrong variable name:", var_name)
  
  # Find columns for that variable
  cols <- grep(var_name, df_names, fixed = T)
  
  # Create and return data frame with site and variable columns
  df <- data.frame(gsub(paste0(".", var_name), "", names(df)[cols]), 
                   unlist(df[date, cols]))
  names(df) <- c("Site", var_name)
  df
}

# Function to get data for each site for all variables at one date
make_date_df <- function(var_names, date = 1) {
  # Make list with a data frame for each real variable for one date
  df_list <- lapply(var_names, make_var_df, date = date)
  
  # Combine the data frames to get all the real variables for each site at one
  # date
  new_df <- df_list[[1]]
  for (i in 2:length(df_list)) {
    new_df <- merge(new_df, df_list[[i]], by = "Site", all = T)
  }
  
  # Add date column after site
  new_df <- cbind(Site = new_df[, 1], Date = date, new_df[, -1])
  
  # Check that no data lost or created.  Cells in new - site and date columns -
  # missing != Cells in old for this date - missing - non-missing skipped vars
  if (prod(dim(new_df)) - 2 * nrow(new_df) - sum(is.na(new_df)) !=
      ncol(df) - sum(is.na(df[date, ])) - sum(skipped) +
      sum(is.na(df[date, skipped]))){
    stop("Data lost or created")
  }
  
  # Return data frame
  new_df
}

# Define variables - skipping "OZONE.1h.pphm" for now as hard to distinguish
var_names <- gsub("1h", "24h", 
                  c("PM10.1h.mcrg.m3", "CO.1h.ppm", 
                    "SO2.1h.pphm", "NO.1h.pphm", "NO2.1h.pphm",
                    "OZONE.1h.pphm.1"))

# Check all variables included except date and "OZONE.1h.pphm"
skipped <- !(1:ncol(df) %in% unlist(sapply(var_names, grep, df_names)))
df_names[skipped]

make_var_df(var_name = var_names[1])
grep("PM10", df_names)
nrow(df)
dim(date_df)
head(date_df)
tail(date_df)

# Number of dates
n_d <- nrow(df)

# Stack data frames over all dates
date_df <- make_date_df(var_names, date = 1)
for (t in 2:n_d) {
  date_df <- rbind(date_df, make_date_df(var_names, date = t))
}
head(date_df, 60)
date_df[date_df$Date == 1, ]
tail(date_df)

# Check that no data lost or created.  Cells in new - site and date columns -
# missing != Cells in old for this date - missing - non-missing skipped vars
if (prod(dim(date_df)) - 2 * nrow(date_df) - sum(is.na(date_df)) !=
        prod(dim(df)) - sum(is.na(df)) - sum(skipped) * nrow(df) +
        sum(is.na(df[, skipped])))
  stop("Data lost or created")

# Read in station information dataset
# list.files()
site_df <- read.csv("Raw sites data.csv", header = T)

# Merge with site data

# Tidy up column names
date_df[, 1] <- gsub(".", " ", date_df[, 1], fixed = T)
site_df[, 2] <- toupper(site_df[, 2])
names(site_df)[c(2, 6:8)] <- c("Site", "Lat.South", "Long.East", "Alt.AHD")

# Stop if any sites in the AQ data have multiple distinct locations in the site
# data
if (any(duplicated(site_df[, 2]) & !duplicated(site_df[, c(2, 6:8)]) & 
        (site_df$Site %in% date_df$Site))) {
  stop("Multiple locations listed for site")
}

# Otherwise just take first location from site data
date_site_df <- merge(site_df[!duplicated(site_df[, 2]), c(2, 6:8)], 
                      date_df, by = "Site")

# Order by date and reset row names
date_site_df <- date_site_df[order(date_site_df[, 5]), ] 
row.names(date_site_df) <- NULL

# Show results
print(head(date_site_df[, 1:6], 18))
print(tail(date_site_df[, 1:6], 18))

# Write to csv
write.csv(
  x = date_site_df, 
  file = "Daily avg air quality data 21-04-2020 to 2021 with site data.csv",
  row.names = F)
