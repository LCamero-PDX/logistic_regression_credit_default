library(corrplot)
library(ggplot2)
library(randomForest)
library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(gbm)
library(MASS)
library(e1071)
library(jtools)
library(ROCR)
library(naivebayes)
library(e1071)
library(jtools)
library(PerformanceAnalytics)
library('OneR')
library('woeBinning')
library(flux)
library(pROC)
library(PRROC)
library(InformationValue)
library(ModelMetrics)
library(dplyr)

############################################################################### LOAD DATA

# set up the file path
my.path <- 'C:\\Users\\lcamero\\Downloads\\';
my.file <- paste(my.path,'credit_card_default.RData',sep='');


# Read the RData object using readRDS();
credit_card_default <- readRDS(my.file)
cc <- credit_card_default

############################################################################### FEATURE ENGINEERING

# Correct categrorical data
cc$EDUCATION <- ifelse(cc$EDUCATION == 0, 4,
                       ifelse(cc$EDUCATION == 5, 4,
                              ifelse(cc$EDUCATION == 6, 4,cc$EDUCATION)))
cc$MARRIAGE <- ifelse(cc$MARRIAGE == 0, 3,cc$MARRIAGE)
cc$PAY_0 <- ifelse(cc$PAY_0 == -2, -1,
                   ifelse(cc$PAY_0 == -1, 0, cc$PAY_0))
cc$PAY_2 <- ifelse(cc$PAY_2 == -2, -1,
                   ifelse(cc$PAY_2 == -1, 0, cc$PAY_2))
cc$PAY_3 <- ifelse(cc$PAY_3 == -2, -1,
                   ifelse(cc$PAY_3 == -1, 0, cc$PAY_3))
cc$PAY_4 <- ifelse(cc$PAY_4 == -2, -1,
                   ifelse(cc$PAY_4 == -1, 0, cc$PAY_4))
cc$PAY_5 <- ifelse(cc$PAY_5 == -2, -1,
                   ifelse(cc$PAY_5 == -1, 0, cc$PAY_5))
cc$PAY_6 <- ifelse(cc$PAY_6 == -2, -1,
                   ifelse(cc$PAY_6 == -1, 0, cc$PAY_6))
colnames(cc)[colnames(cc)=="PAY_0"] <- "PAY_1"

############################################################################### BINNING MODEL

#rewrite AGE with new bins
cc$age_bin <- ifelse(cc$AGE <= 25, 25,
                     ifelse(cc$AGE <= 35, 35,
                            ifelse(cc$AGE <= 45, 45, 55)))

############################################################################### CALCULATE FEATURES

# average bill amount over 6 months
cc$avg_bill_amt <- (cc$BILL_AMT1 + cc$BILL_AMT2 + cc$BILL_AMT3 + cc$BILL_AMT4 + cc$BILL_AMT5 + cc$BILL_AMT6)/6 

# average payment amount
cc$avg_pmt_amt <- (cc$PAY_AMT1 + cc$PAY_AMT2 + cc$PAY_AMT3 + cc$PAY_AMT4 + cc$PAY_AMT5 + cc$PAY_AMT6)/6 

# Payment Ratio
cc$pmt_ratio1 <- cc$PAY_AMT1/(ifelse(cc$BILL_AMT2 == 0,1,cc$BILL_AMT2))
cc$pmt_ratio2 <- cc$PAY_AMT2/(ifelse(cc$BILL_AMT3 == 0,1,cc$BILL_AMT3))
cc$pmt_ratio3 <- cc$PAY_AMT3/(ifelse(cc$BILL_AMT4 == 0,1,cc$BILL_AMT4))
cc$pmt_ratio4 <- cc$PAY_AMT4/(ifelse(cc$BILL_AMT5 == 0,1,cc$BILL_AMT5))
cc$pmt_ratio5 <- cc$PAY_AMT5/(ifelse(cc$BILL_AMT6 == 0,1,cc$BILL_AMT6))

# Average Payment Ratio
cc$avg_pmt_ratio <- (cc$pmt_ratio1 + cc$pmt_ratio2 + cc$pmt_ratio3 + cc$pmt_ratio4 + cc$pmt_ratio5)/5

# fix payment ratio nulls to 100
cc[is.na(cc)] <- 100

# max bill amount
cc$max_bill_amt <- pmax(cc$BILL_AMT1,cc$BILL_AMT2,cc$BILL_AMT3,cc$BILL_AMT4,cc$BILL_AMT5,cc$BILL_AMT6) 

# max payment amount
cc$max_pmt_amt <- pmax(cc$PAY_AMT1,cc$PAY_AMT2,cc$PAY_AMT3,cc$PAY_AMT4,cc$PAY_AMT5,cc$PAY_AMT6)

# Utilization
cc$util <- cc$BILL_AMT1/cc$LIMIT_BAL
cc$util2 <- cc$BILL_AMT2/cc$LIMIT_BAL
cc$util3 <- cc$BILL_AMT3/cc$LIMIT_BAL
cc$util4 <- cc$BILL_AMT4/cc$LIMIT_BAL
cc$util5 <- cc$BILL_AMT5/cc$LIMIT_BAL
cc$util6 <- cc$BILL_AMT6/cc$LIMIT_BAL

# average utilization
cc$avg_util <- (cc$util + cc$util2 + cc$util3 + cc$util4 + cc$util5 + cc$util6)/6

# balance growth over 6 months and convert to binary
cc$bal_growth_6mo <- cc$BILL_AMT6 > cc$BILL_AMT1
cc$bal_growth_6mo <- 1 * cc$bal_growth_6mo

# utilization growth over 6 months and convert ot binary
cc$util_growth_6mo <- cc$util6 > cc$util
cc$util_growth_6mo <- 1 * cc$util_growth_6mo

# max delinquency
cc$max_DLQ <- pmax(cc$PAY_1,cc$PAY_2,cc$PAY_3,cc$PAY_4,cc$PAY_5,cc$PAY_6) 

# scale the utilization
summary(cc$util)
cc$util <- cc$util*100
cc$util2 <- cc$util2*100
cc$util3 <- cc$util3*100
cc$util4 <- cc$util4*100
cc$util5 <- cc$util5*100
cc$util6 <- cc$util6*100

# calculate minimum payment ratio
cc$min_pmt_ratio <- pmin(cc$pmt_ratio1, cc$pmt_ratio2, cc$pmt_ratio3, cc$pmt_ratio4, cc$pmt_ratio5)

# Create calculated field based on the interaction between age and education
cc$education_by_age <- cc$EDUCATION*cc$AGE

############################################################################### TRANSFORM VARIABLES

# transform the variables into the log
cc$AGE_log <- log(cc$AGE)

############################################################################### VISUALIZE VARIABLES

sum(cc$train)
sum(cc$test)
sum(cc$validate)

############################################################################### DISCRETE BINNING

# oneR binning for Limit Balance
cc$LIMiT_BAL_bin <- ifelse(cc$LIMIT_BAL >= 130000,1,0)
cc$DEFAULT <- as.numeric(cc$DEFAULT)

############################################################################### MODEL

# pick the model that is only the training data set 
cc_train <- subset(cc, train == 1)
cc_validate <- subset(cc, validate == 1)
cc_test <- subset(cc, test == 1)

############################################################################### LOGISTIC REGRESSION

# use variable selection method: stepwise
# Define the upper model as the FULL model
upper.lm <- glm(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE 
                + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6
                + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5
                + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4
                + PAY_AMT5 + PAY_AMT6 + age_bin + avg_bill_amt + pmt_ratio1 
                + pmt_ratio2 + pmt_ratio3 + pmt_ratio4 + pmt_ratio5 + avg_pmt_ratio 
                + max_bill_amt + max_pmt_amt + util + util2 + util3 + util4 + util5 + util6 + avg_util
                + bal_growth_6mo + util_growth_6mo + max_DLQ + min_pmt_ratio + education_by_age + AGE_log 
                , data = cc_train, family = binomial());

# check the full model results
summary(upper.lm)

# Define the lower model as the Intercept model
lower.lm <- glm(DEFAULT ~ 1,data = cc_train, family = binomial());

# check the summary of the lower model
summary(lower.lm)

# create backward model
backward.lm <- stepAIC(object=upper.lm,direction=c('backward'), family = binomial());
summary(backward.lm)

cc_train$DEFAULT_backward <- predict(backward.lm, type = "response")

AIC(backward.lm)
BIC(backward.lm)
print(-2*logLik(backward.lm, REML = TRUE))
mae(cc_train$DEFAULT, cc_train$DEFAULT_backward)
ks_stat(actuals=cc_train$DEFAULT, predictedScores=cc_train$DEFAULT_backward)

summary(backward.lm)

############################################################################### LOGISTIC RESULTS

library('ROCR')

# AUC function
fun.auc <- function(pred,obs){
  # Run the ROCR functions for AUC calculation
  ROC_perf <- performance(prediction(pred,obs),"tpr","fpr")
  ROC_sens <- performance(prediction(pred,obs),"sens","spec")
  ROC_err <- performance(prediction(pred, labels=obs),"err")
  ROC_auc <- performance(prediction(pred,obs),"auc")
  # AUC value
  AUC <- ROC_auc@y.values[[1]] # AUC
  # Mean sensitivity across all cutoffs
  x.Sens <- mean(as.data.frame(ROC_sens@y.values)[,1])
  # Mean specificity across all cutoffs
  x.Spec <- mean(as.data.frame(ROC_sens@x.values)[,1])
  # Sens-Spec table to estimate threshold cutoffs
  SS <- data.frame(SENS=as.data.frame(ROC_sens@y.values)[,1],SPEC=as.data.frame(ROC_sens@x.values)[,1])
  # Threshold cutoff with min difference between Sens and Spec
  SS_min_dif <- ROC_perf@alpha.values[[1]][which.min(abs(SS$SENS-SS$SPEC))]
  # Threshold cutoff with max sum of Sens and Spec
  SS_max_sum <- ROC_perf@alpha.values[[1]][which.max(rowSums(SS[c("SENS","SPEC")]))]
  # Min error rate
  Min_Err <- min(ROC_err@y.values[[1]])
  # Threshold cutoff resulting in min error rate
  Min_Err_Cut <- ROC_err@x.values[[1]][which(ROC_err@y.values[[1]]==Min_Err)][1]
  # Kick out the values
  round(cbind(AUC,x.Sens,x.Spec,SS_min_dif,SS_max_sum,Min_Err,Min_Err_Cut),3)
}

# Run the function with the example data
observations <- as.numeric(cc_validate$DEFAULT)
predictions <- as.numeric(predict(backward.lm, newdata = cc_validate,  type = "response"))
fun.auc(predictions, observations)

cc_validate$DEFAULT_backward <- predict(backward.lm, newdata = cc_validate,  type = "response")
cc_validate$DEFAULT_backward_ind <- ifelse(cc_validate$DEFAULT_backward >= 0.5, 1,0)
summary(cc_validate$DEFAULT_backward)
one <- cc_validate %>%  filter(DEFAULT == 1)
zero <- cc_validate %>%  filter(DEFAULT != 1)
table(one$DEFAULT_backward_ind)
table(zero$DEFAULT_backward_ind)

############################################################################### ROC

library(pROC)
g <- roc(DEFAULT ~ DEFAULT_backward, data = cc_validate)
plot(g, col = "dark green")   

g <- roc(DEFAULT ~ DEFAULT_backward, data = cc_validate)
plot(g, col = "dark green")  

write.csv(cc_validate, file = "lift_data_validation.csv")

############################################################################### KS Statistic

table(cc_validate$DEFAULT)

decile.pts <- quantile(cc_validate$DEFAULT_backward,
                       probs=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9));

cc_validate$model.decile <- cut(cc_validate$DEFAULT_backward,breaks=c(0,decile.pts,1),
                          labels=rev(c('01','02','03','04','05','06','07','08','09','10'))
);

head(cc_validate)

# Note that we want the 'top decile' to be the highest model scores so we
# will reverse the labels.

# Check the min score in each model decile;
aggregate(cc_validate$DEFAULT_backward,by=list(Decile=cc_validate$model.decile),FUN=mean);

table(cc_validate$model.decile)

cc_validate$response <- ifelse(cc_validate$DEFAULT_backward < .5,1,0)

table(cc_validate$model.decile,cc_validate$DEFAULT)

ks.table <- as.data.frame(list(Y0=table(cc_validate$model.decile,cc_validate$DEFAULT)[,1],
                               Y1=table(cc_validate$model.decile,cc_validate$DEFAULT)[,2],
                               Decile=rev(c('01','02','03','04','05','06','07','08','09','10'))
));


# Sort the data frame by decile;
ks.table[order(ks.table$Decile),]