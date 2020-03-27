# Patrick Youssef
# 87834207
# Solution to MAE 195 Final Project

#==========================================================================
# Import Packages ---------------------------------------------------------
#==========================================================================

library(dplyr)
library(plyr)
library(tidyr)
library(tictoc)
library(tidyverse)
library(e1071)
library(pracma)
library(caTools)
library(ggplot2)
library(caret)
library(randomForest)
library(Amelia)
library(RWeka)
library(parallel)
library(doParallel)

#==========================================================================
# Data Import -------------------------------------------------------------
#==========================================================================

# This sections imports all the data into one large df_all dataframe
# This is one ran once and all feature engineering is done later on the
# large dataframe.

# Move working directory to the folder containing the dataset
setwd("~/Documents/ML_Data/wisdm-dataset/")

# Narrow in on raw data and a particular user, device, and sensor
users <- 1600:1650
users <- users[users != 1614]
actions <- toupper(letters)[1:19]
actions <- actions[-14]
row_cut <- 17

df_PA <- data.frame()
df_PG <- data.frame()
df_WA <- data.frame()
df_WG <- data.frame()

for (user in users) {
  print(sprintf("User %i started importing", user))
  pattern_string <- sprintf("data_%i_.*_.*.arff", user)
  files <- list.files(path = getwd(), recursive = T, pattern = pattern_string)
  
  PA <- cbind(data.frame(User = user), read.arff(files[1]))
  PG <- cbind(data.frame(User = user), read.arff(files[2]))
  WA <- cbind(data.frame(User = user), read.arff(files[3]))
  WG <- cbind(data.frame(User = user), read.arff(files[4]))
  
  # Find number of columns for later
  n_c <- ncol(PA)
  
  # File list
  file_list <- list(PA, PG, WA, WG)
  
  # Get actions for the users
  missing_actions = NULL
  for (i in 1:4) {
    if (length(unique(data.frame(file_list[i])$ACTIVITY)) != 18){
      missing_actions <- append(missing_actions, actions[!(actions %in% data.frame(file_list[i])$ACTIVITY)])
    }
  }
  missing_actions <- unique(missing_actions)
  user_actions <- actions[!(actions %in% missing_actions)]
  
  # Re-Init dataframes to increase speed of action concatentation
  df_PA_action <- data.frame()
  df_PG_action <- data.frame()
  df_WA_action <- data.frame()
  df_WG_action <- data.frame()
  
  for (action in user_actions) {
    # Add entries to a list of this user's data
    df_PA_action <- rbind(df_PA_action, PA[PA$ACTIVITY == action,][1:row_cut, -n_c])
    df_PG_action <- rbind(df_PG_action, PG[PG$ACTIVITY == action,][1:row_cut, -n_c])
    df_WA_action <- rbind(df_WA_action, WA[WA$ACTIVITY == action,][1:row_cut, -n_c])
    df_WG_action <- rbind(df_WG_action, WG[WG$ACTIVITY == action,][1:row_cut, -n_c])
  }
  
  df_PA <- rbind(df_PA, df_PA_action)
  df_PG <- rbind(df_PG, df_PG_action)
  df_WA <- rbind(df_WA, df_WA_action)
  df_WG <- rbind(df_WG, df_WG_action)
  
}

users_df <- unique(df_PA$User)

# Bind all data together
df_all <- bind_rows("PA" = df_PA, "PG" = df_PG, "WA" = df_WA, "WG" = df_WG, .id = "Sensor")
df_all <- df_all[order(df_all$User, df_all$ACTIVITY),]
rownames(df_all) <- seq(length=nrow(df_all))

# Move working directory to the main folder
setwd("~/Dropbox/MAE_195_ML/Projects/Final Project/")

# Save dataframes as RDS
saveRDS(df_all, file = "df_ARFF.rds")

#==========================================================================
# Feature Reorginization --------------------------------------------------
#==========================================================================

# Load dataframe (assuming that the prior step was done at a different time)
setwd("~/Dropbox/MAE_195_ML/Projects/Final Project/")
if (!exists("df_all")){
  df_all <- readRDS('df_ARFF.rds')
}

# Refactor rows for a particular action into one column
df_features_clean <- data.frame()
new_names <- c('User', 'Action')
names <- names(df_all)[-c(1,2,3)]

# Create new column names
for (p in c('PA', 'PG', 'WA', 'WG')) {
  new_names <- append(new_names, paste(sep = '_', p, names))
}

for (i in 1:(nrow(df_all) / (4*17))) {
  df_features_clean <- rbind(df_features_clean, cbind(df_all[((i-1)*4*17+1:17), -c(1)], 
                                                      df_all[((i-1)*4*17+1:17+17), -c(1,2,3)],
                                                      df_all[((i-1)*4*17+1:17+34), -c(1,2,3)], 
                                                      df_all[((i-1)*4*17+1:17+51), -c(1,2,3)]))
}
names(df_features_clean) <- new_names

# Check for zero columns
c_z <- colSums(df_features_clean[,-c(1,2)]) != 0
c_z <- append(c(T, T), c_z)
df_features_clean <- df_features_clean[,c_z]

# Standardize the data
df_features_clean <- cbind(df_features_clean[,1:2], data.frame(scale(df_features_clean[, -c(1:2)], 
                                                                     center = TRUE, scale = TRUE)))

# Remove highly-correlated features
correlationMatrix <- cor(df_features_clean[,3:ncol(df_features_clean)])
correlationMatrix[is.na(correlationMatrix)] = 0
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.85)
df_features_clean <- df_features_clean[,!(1:ncol(df_features_clean) %in% highlyCorrelated)]

#==========================================================================
# Test/Train Split --------------------------------------------------------
#==========================================================================

# Now that we have a dataframe of the features we need to split it
# Splits cannot be based on users as not all users have all actions
# Because of this, I will split at the action level and regroup it after

# Import feature data set
df_features <- readRDS('df_features.rds')
actions <- toupper(letters)[1:19]
actions <- actions[-14]

# 70% of the data will be used to train
split_ratio = 0.70

# Set a seed for repeatable results
set.seed(100)

# Create empty data frame for train and test and split
df_train = data.frame()
df_test = data.frame()
df <- df_features

users <- unique(df$User)
n = length(users)
inds <- sample.split(1:n, SplitRatio = split_ratio)
users_train <- users[inds]
users_test <- users[!inds]
df_train <- bind_rows(df_train, df[df$User %in% users_train,])
df_test <- bind_rows(df_test, df[df$User %in% users_test,])

# Save dataframes as RDS
saveRDS(df_train, file = "df_train.rds")
saveRDS(df_test, file = "df_test.rds")

# Model Analysis ----------------------------------------------------------

# Run algorithms using 5-fold cross validation
control <- trainControl(method="cv", number=5, allowParallel=T)
metric <- "Accuracy"
cluster <- makeCluster(detectCores() - 2) # convention to leave 2 threads for OS
registerDoParallel(cluster)

# Set up data
data_all <- subset(df_features, select = -c(User))
data_train <- subset(df_train, select = -c(User))
data_test <- subset(df_test, select = -c(User))

# kNN
set.seed(7)
print('KNN Train Begin')
fit.knn <- train(Action ~ ., data=data_train, method="knn",
                 metric=metric, trControl=control)
pred.knn <- predict(fit.knn, data_test)
conf.knn <- confusionMatrix(pred.knn, data_test$Action)

# SVM
set.seed(7)
print('SVM Train Begin')
fit.svm <- train(Action ~ ., data=data_train, method="svmRadial", 
                 metric=metric, trControl=control)
pred.svm <- predict(fit.svm, data_test)
conf.svm <- confusionMatrix(pred.svm, data_test$Action)

# Random Forest
set.seed(7)
print('RF Train Begin')
fit.rf <- train(Action ~ ., data=data_train, method="rf", 
                metric=metric, trControl=control)
pred.rf <- predict(fit.rf, data_test)
conf.rf <- confusionMatrix(pred.rf, data_test$Action)

#summarize accuracy of models
results <- resamples(list(knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# Binned Accuracy ---------------------------------------------------

bin_accuracy <- function(pred_samp, df_test) {
  probs <- data.frame(df_test[,c(1,2)])
  probs <- probs[!duplicated(probs),]
  names <- colnames(probs)
  probs <- cbind(probs, as.data.frame(matrix(0, ncol = 18, nrow = nrow(probs))))
  colnames(probs) <- append(names, actions)
  
  # Check for real accuracy
  for (i in 1:nrow(probs)) {
    for (k in 1:17) {
      pred <- pred_samp[(i-1)*17 + k]
      probs[i,-c(1,2)] = probs[i,-c(1,2)] + as.integer(pred == actions)
    }
  }
  probs$WeightedPred <- actions[max.col(probs[,-c(1,2)])]
  accuracy_weighted <- sum(probs$Action == probs$WeightedPred) / nrow(probs)
  conf <- confusionMatrix(factor(probs$Action), factor(probs$WeightedPred))
  results <- list(accuracy_weighted, probs, conf)
  names(results) <- c('Accuracy', 'VoteTable', 'Confusion')
  return(results)
}

# Check for binned accuracy
acc.knn <- bin_accuracy(pred.knn, df_test)
acc.svm <- bin_accuracy(pred.svm, df_test)
acc.rf <- bin_accuracy(pred.rf, df_test)

# Table of results
df_results <- rbind(fit.knn$resample, fit.svm$resample, fit.rf$resample)
df_results <- cbind(data.frame(Resample = df_results$Resample), df_results[,-c(3)])
df_results <- cbind(data.frame(Model = c(rep('KNN', 5), rep('SVM', 5), rep('RF', 5))), df_results)
df_results <- cbind(df_results, data.frame('Accuracy Test' = c(rep(conf.knn$overall[1], 5), rep(conf.svm$overall[1], 5), 
                                                                 rep(conf.rf$overall[1], 5))))
df_results <- cbind(df_results, data.frame('Accuracy Binned' = c(rep(acc.knn$Accuracy, 5), rep(acc.svm$Accuracy, 5), 
                                        rep(acc.rf$Accuracy, 5))))

df_action <- data.frame(matrix(0, ncol = 18, nrow = 1))
colnames(df_action) <- actions 
df_action[1,] <- c('Walking', 'Jogging', 'Stairs', 'Sitting', 'Standing', 'Typing',
                   'Teeth', 'Soup', 'Chips', 'Pasta', 'Drinking', 'Sandwich',
                   'Kicking', 'Catch', 'Dribbling', 'Writing',
                   'Clapping', 'Folding')
