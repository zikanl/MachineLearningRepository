---
title: "Coursera - Practical Machine Learning - Course Project Writeup"
author: "Miroslav Zivkovic"
--- 

Last updated 2015-08-24 00:30:11 using R version 3.2.1 (2015-06-18).
 
## Executive Summary  

This report shows the results of applying a variety of machine learning algorithms to the Weight Lifting Exercise Dataset, 
originally available from http://groupware.les.inf.puc-rio.br/har.    

A short explanation is given how the model was built, the cross validation, and out-of-sample error rate.
     
The following algorithms were applied using the train() function in the caret package:  

* Tree (rpart)  
* Random Forest (rf)
* Boosting (gbm)  
* Linear Discriminant Analysis (lda)  
* Naive Bayesian (nb)  

The best results were obtained using the random forest algorithm. 
Predicting with the testing data set and the random forest model yields the following results:

* Estimated accuracy of 99.34% (95% CI of 99.13% to 99.51%)
* Implying an out-of-sample error rate of 0.66% (95% CI of 0.49% to 0.87%) 

Overfitting was minimized by cleaning the data carefully and by using cross validation in the train() function.  

## Analysis  

### R Packages and Settings  


```r
# global settings for knitr
library(knitr)
opts_chunk$set(message=FALSE,
               warnings=FALSE,
               tidy=TRUE,
               echo=FALSE,
               fig.height=3,
               fig.width=4)
# load required libraries
library(caret)
library(downloader)
library(randomForest)
```

```
## Warning: package 'downloader' was built under R version 3.2.2
```

### Question

A total of six participants participated in a lifting exercise in five different ways. The five ways, were exactly according to the 
specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) 
and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.

By processing data gathered from accelerometers on the belt, forearm, arm, and dumbell of the participants in a machine learning algorithm, the question is can the appropriate 
activity quality (class A-E) be predicted?

### Data Preparation  
Significant data cleansing was required.  The code comments that explain what has been done are shown below.  
It is assumed that the data both training and testing data sets have been downloaded and placed in the same
directory as script.

```r
# download the files
trainfile <- "pml-training.csv"
testfile <- "pml-testing.csv"

# Set URL from which data is downloaded
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download(trainURL, trainfile)
download(testURL, testfile)

# read files from the same directory as script
pml_train <- read.csv(trainfile, header = TRUE, na.strings = c("", "NA"))
validation <- read.csv(testfile, header = TRUE, na.strings = c("", "NA"))

# Verifies that the column names are identical in the training and test set.
all.equal(colnames(pml_train)[1:length(colnames(pml_train)) - 1], colnames(validation)[1:length(colnames(validation)) - 
    1])
```

```
## [1] TRUE
```

```r
# partition pml_train into training and testing data sets
set.seed(32343)
inTrain <- createDataPartition(y = pml_train$classe, p = 0.6, list = FALSE)
training <- pml_train[inTrain, ]
testing <- pml_train[-inTrain, ]

# filter out new_window rows from all three data sets
training <- training[training$new_window != "yes", ]
testing <- testing[testing$new_window != "yes", ]
validation <- validation[validation$new_window != "yes", ]

# filter out covariates with near zero variance; most values in these
# columns are NA's; use training data to determine which covariates will be
# filtered out
skip_columns <- nearZeroVar(training)
training <- training[, -skip_columns]
testing <- testing[, -skip_columns]
validation <- validation[, -skip_columns]

# remove index number, subject name, and timestamp columms; instructions for
# project specifically say use the accelerometer data (only); including
# these columns would contribute to overfitting.
omit_columns <- c(1:6)
training <- training[, -omit_columns]
testing <- testing[, -omit_columns]
validation <- validation[, -omit_columns]

# split data sets into predictor vectors and outcome vector; this is a
# recommended optimization for train() method
training_predictors <- training[, -53]
training_outcome <- training[, 53]
```
### Random Forest  

```r
# optimal mtry parameter value was obtained from previous run of the model;
# saves time to just pass it in on subsequent runs
set.seed(32343)
mtryGrid <- expand.grid(mtry = 2)
rf <- train(x = training_predictors, y = training_outcome, method = "rf", metric = "Accuracy", 
    trControl = trainControl(method = "cv", repeats = 5), tuneGrid = mtryGrid, 
    prox = TRUE)
rf
```

```
## Random Forest 
## 
## 11528 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 10375, 10375, 10376, 10375, 10375, 10376, ... 
## 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD   
##   0.9913249  0.9890249  0.002350955  0.002974092
## 
## Tuning parameter 'mtry' was held constant at a value of 2
## 
```

```r
# show confusion matrix for testing data only
pred <- predict(rf, testing)
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2190    8    0    0    0
##          B    2 1481   15    0    0
##          C    0    3 1322   16    0
##          D    0    0    5 1240    1
##          E    0    0    0    1 1404
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9934          
##                  95% CI : (0.9913, 0.9951)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9916          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9926   0.9851   0.9865   0.9993
## Specificity            0.9985   0.9973   0.9970   0.9991   0.9998
## Pos Pred Value         0.9964   0.9887   0.9858   0.9952   0.9993
## Neg Pred Value         0.9996   0.9982   0.9968   0.9974   0.9998
## Prevalence             0.2851   0.1941   0.1746   0.1635   0.1828
## Detection Rate         0.2849   0.1926   0.1720   0.1613   0.1826
## Detection Prevalence   0.2859   0.1948   0.1744   0.1621   0.1828
## Balanced Accuracy      0.9988   0.9949   0.9911   0.9928   0.9996
```

## Appendix
The following code was used to output the results of validation testing for submitting to Coursera.  

```r
# file writing function
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}
pred2 <- as.character(predict(rf, validation))
pml_write_files(pred2)
```
