getwd()
setwd("/Users/wooyongjung/OneDrive/Spring 2020/MachineLearning/MLinEcon2020/ridge_lasso")

# Load required packages
install.packages("glmnet")
library(glmnet)
set.seed(42)

# Set the number of observation (n = 1000) and features (p = 5000).
n <- 1000
p <- 5000

# Set the number of the meaningful features (real_p = 15)
real_p <- 15

# Create a matrix (n x p)
x <- matrix(rnorm(n*p), nrow = n, ncol = p)

# Create y using only the first 15 features (real_p) with some noise
y <- apply(x[,1:real_p], 1, sum) + rnorm(n)
length(y)

# Create a train data index and seperate the data into train and test set.
train_rows <- sample(1:n, .66*n) 
x.train <- x[train_rows,]
x.test <- x[-train_rows,]
y.train <- y[train_rows]
y.test <- y[-train_rows]

# Ridge regression
alpha0.fit <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = 0, family = "gaussian")
alpha0.predicted <- predict(alpha0.fit, s = alpha0.fit$lambda.1se, newx = x.test)
mean((y.test - alpha0.predicted)^2)

# Lasso regression
alpha1.fit <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = 1, family = "gaussian")
alpha1.predicted <- predict(alpha1.fit, s = alpha1.fit$lambda.1se, newx = x.test)
mean((y.test - alpha1.predicted)^2)

# Elastic-net
alpha05.fit <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = .5, family = "gaussian")
alpha05.predicted <- predict(alpha05.fit, s = alpha05.fit$lambda.1se, newx = x.test)
mean((y.test - alpha05.predicted)^2)

# Cross-validation to find lambda
list.of.fits <- list()
for (i in 0:10) {
  fit.name <- paste0("alpha", i/10)
  list.of.fits[[fit.name]] <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = i/10, family = "gaussian")
}
list.of.fits[1]

results <- data.frame()
for (i in 0:10) {
  fit.name <- paste0("alpha", i/10)
  predicted <- predict(list.of.fits[[fit.name]], s = list.of.fits[[fit.name]]$lambda.1se, newx = x.test)
  mse <- mean((y.test - predicted)^2)
  temp <- data.frame(alpha = i/10, mse = mse, fit.name = fit.name)
  results <- rbind(results, temp)
}

# Test git push


paste0(c("A","B","C"), "&", 1:3, collapse= "")

