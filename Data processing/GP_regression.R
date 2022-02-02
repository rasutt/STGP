###
##Loading data
data <- read.csv("Daily avg air quality data 21-04-2020 to 2021 with site data.csv"
                 , header = T)

head(data)
n <- dim(data)[1]
m <- dim(data)[2]
apply(data,2,function(x) sum(is.na(x))/n)


### data processing

length(unique(data$Site))
length(unique(data$Date))
## Number of different sites
# 44

## Number of different sites
# 366


mis_vals_sites <- aggregate(is.na(NO2.24h.pphm) ~ Site, data = data, sum)
colnames(mis_vals_sites) <- c("Site", "NA_Count")
(mis_vals_sites <- mis_vals_sites[order(mis_vals_sites$NA_Count, decreasing = T),])

##which mean ALBURY, BATHURST, LINDFIELD, LIVERPOOL SWAQS, MACARTHUR, ORANGE, TAMWORTH, VINEYARD are invalid sites
invalid_sites <- c("ALBURY", "BATHURST", "LINDFIELD", "LIVERPOOL SWAQS"
                   , "MACARTHUR", "ORANGE", "TAMWORTH", "VINEYARD"
                   , "KATOOMBA", "MORISSET")
data <- subset(data, !(Site %in% invalid_sites))
mis_vals_sites <- aggregate(is.na(NO2.24h.pphm) ~ Site, data = data, sum)
colnames(mis_vals_sites) <- c("Site", "NA_Count")
(mis_vals_sites <- mis_vals_sites[order(mis_vals_sites$NA_Count, decreasing = T),])
length(unique(data$Site)) ## new number of sites = 34


mis_vals_dates <- aggregate(is.na(NO2.24h.pphm) ~ Date, data = data, sum)
colnames(mis_vals_dates) <- c("Date", "NA_Count")
(mis_vals_dates <- mis_vals_dates[order(mis_vals_dates$NA_Count, decreasing = T),])
#date == 366 are all null values
data <- subset(data, Date < 366)

write.table(data, "Filtered_Data_Paco_04052021.csv", sep = ", ")

##loading libraries
require(ggplot2)
require(GPfit)
require(tidyverse)
require(rayshader)
require(hexbin)
require(magick)


#############
##scaling algorithm
minmax_norm <- function(values) (values - min(values))/(max(values) - min(values))

#####Temporal GP


data_liverpool <- subset(data, Site == 'LIVERPOOL' & !is.na(NO2.24h.pphm), select = c(Date,NO2.24h.pphm))
data_liverpool['Date'] <- minmax_norm(data_liverpool$Date)

colnames(data_liverpool) <- c('X', 'Y')
liverpool <- data_liverpool %>% as.matrix()
plot(Y ~ X, data = liverpool, typ = 'l')

sd(data_liverpool$Y)

#Fitting GP
liverpool_gp <- GP_fit(X = liverpool[,'X']
                       , Y = liverpool[,'Y']
                       , corr = list(type = "exponential", power = 1.95))


##predicting new values
x_new <- seq(0, 1, length.out = 1000)
pred <- predict.GP(liverpool_gp, xnew = data.frame(x = x_new))
mu <- pred$Y_hat
sigma <- sqrt(pred$MSE)
sigma <- sigma/sd(data_liverpool$Y)
sigma <- 10*sigma
####plotting GP
ggplot(as.data.frame(liverpool))+
  geom_line(data = data.frame(x = x_new, y = mu),
            aes(x = x, y = y), color = "red", linetype = "dashed")+
  geom_ribbon(data = data.frame(x = x_new, y_up = mu + sigma, y_low = mu - sigma), 
              aes(x = x_new, ymax = y_up, ymin = y_low), fill = "skyblue", alpha = 0.5) +
  geom_point(data = data_liverpool, aes(x = X,y = Y), size = 2)+
  theme_minimal() +
  labs(title = "Gaussian Process Posterior of f(x)",
       subtitle = "Blue area indicate the credible intervals",
       y = "f(x)")

####for spatial GP
data103 <- subset(data, Date ==103, select = c(Site, Lat.South, Long.East, NO2.24h.pphm))
head(data103)
range(data103$NO2.24h.pphm)
data103$Lat.South_scale <- minmax_norm(data103$Lat.South)
data103$Long.East_scale <- minmax_norm(data103$Long.East)


### plotting original NO2 on date =103
data103_pp = ggplot(data103) + 
  geom_point(aes(x=Lat.South,color=NO2.24h.pphm,y=Long.East),size=2) +
  scale_color_continuous(limits=c(0,3)) +
  ggtitle("Date 103") +
  theme(title = element_text(size=8),
        text = element_text(size=12)) 

plot_gg(data103_pp, height=3, width=3.5, multicore=TRUE, pointcontract = 0.7, soliddepth=-200)
render_snapshot(clear = T)


data103[,c("Long.East_scale", "Lat.South_scale")]


spatial_GP_103 <- GP_fit(X = data103[,c("Lat.South_scale", "Long.East_scale")]
                        , Y = data103[,"NO2.24h.pphm"]
                        , corr = list(type = "exponential", power = 2))

summary(spatial_GP_103)

X_new

data103[,c("Long.East_scale","Lat.South_scale", "NO2.24h.pphm")]
?curve
plot(Lat.South_scale~Long.East_scale, data = data103, pch = 17)

lm1 <- lm(Lat.South_scale~Long.East_scale, data = data103)
x_new <- data.frame(Long.East_scale = seq(0,1, length.out = 100))
pred <- predict.lm(lm1, newdata = x_new)
points(pred ~ seq(0,1, length.out = 100), add = T, col = 'red')


X_new <- sapply(data103$Long.East_scale, function(x) seq(from = 0, to = x, length.out = 10))
(X_new <- as.vector(matrix(X_new, ncol = 1)))
X_new <- cbind(Long.East_scale = X_new, Lat.South_scale = rep(data103$Lat.South_scale, rep(10,34)))
head(X_new)
pred103 <- predict.GP(spatial_GP_103, xnew = X_new)
mu103 <- pred103$Y_hat
range(mu103)
dim(X_new)

mu103
head(X_new)




tp_plot <- rbind(data.frame(cbind(X_new, NO2.24h.pphm = mu103)), data103[,c("Long.East_scale","Lat.South_scale", "NO2.24h.pphm")])

data103_pp = ggplot(tp_plot) + 
  geom_point(aes(x=Long.East_scale,color=NO2.24h.pphm,y=Lat.South_scale),size=2) +
  scale_color_continuous(limits=c(0.1,3)) +
  ggtitle("Date 103") +
  theme(title = element_text(size=8),
        text = element_text(size=12)) 

plot_gg(data103_pp, height=10, width=10, multicore=TRUE, pointcontract = 0.7, soliddepth=-200)
render_snapshot(clear = T)

vals <- seq(0,1, length.out = 30)
x_new <- rep(vals, 30)
y_new <- rep(vals, rep(30,30))
X_new <- data.frame(Long.East_scale = x_new, Lat.South_scale = y_new)
(b = lm1$coefficients[1])
(m = lm1$coefficients[2])
X_new <- subset(X_new, Lat.South_scale >= b + m*Long.East_scale)

plot(x_new, y_new)


#################
###Generating new coordinates for predictions
sites_locations <- data103[,c('Long.East_scale', 'Lat.South_scale')]



pred_site_locations <- apply(sites_locations, 1, function(x) x + mvrnorm(n = 8,mu= c(0,0), Sigma = diag(0.25,nrow =2)))

as.matrix(pred_site_locations , ncol = 2)
str(pred_site_locations)




