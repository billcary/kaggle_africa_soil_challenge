## =============================================================================
## R + caret + Domino Starter Code
## =============================================================================

## Data Source: https://www.kaggle.com/c/afsis-soil-properties


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Load Libraries
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(caret)
library(RSNNS)


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Import Data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Get the local path on Domino
path_cloud <- getwd()

## Define other paths
path_train <- paste0(path_cloud, "/data/train.zip")
path_test <- paste0(path_cloud, "/data/test.zip")
path_submission <- paste0(path_cloud, "/data/sample_submission.csv")
path_output <- paste0(path_cloud, "/results/my_Kaggle_submission.csv")

## Read data files
raw_sub <- read.csv(path_submission)
train_hex <- read.table(unz(path_train, "training.csv"), header=T, quote="\"", sep=",")
test_hex <- read.table(unz(path_test, "sorted_test.csv"), header=T, quote="\"", sep=",")


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train a Deep Neural Networks model for each variable
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Split the dataset into 80:20 for training and validation
## train_hex_split <- h2o.splitFrame(train_hex, ratios = 0.8, shuffle = TRUE)

## Set up training control parameters
tc <- trainControl(method = "cv", number=5, classProbs=FALSE,
                   savePred=T)

## One Variable at at Time
ls_label <- c("Ca", "P", "pH", "SOC", "Sand")


for (n_label in 1:5) {
        
        ## Display
        cat("\n\nNow training a DNN model for", ls_label[n_label], "...\n")
        
        ## Train a DNN
        model <- train(x = train_hex[, 2:3595],
                       y = train_hex[, (3595 + n_label)],
                       method = 'mlp',
                       size = 3,
                       trControl = tc,
                       metric = 'RMSE')
        
        ## Print the Model Summary
        print(model)
        
        ## Use the model for prediction and store the results in submission template
        raw_sub[, (n_label + 1)] <- as.matrix(predict(model, test_hex))
        
}


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Save the results as a CSV
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

write.csv(raw_sub, file = path_output, row.names = FALSE)


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Print System and Session Info
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(sessionInfo())

print(Sys.info())
