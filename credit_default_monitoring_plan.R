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

# run logistic regression
logis <- glm(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE 
             + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6
             + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5
             + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4
             + PAY_AMT5 + PAY_AMT6 + age_bin + avg_bill_amt + pmt_ratio1 
             + pmt_ratio2 + pmt_ratio3 + pmt_ratio4 + pmt_ratio5 + avg_pmt_ratio 
             + max_bill_amt + max_pmt_amt + util + util2 + util3 + util4 + util5 + util6 + avg_util
             + bal_growth_6mo + util_growth_6mo + max_DLQ + min_pmt_ratio + education_by_age + AGE_log 
             , data = cc_train, family = binomial())

# check logistic regression results of the full model
summary(logis)

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

# run stepwise model regression
forward.lm <- stepAIC(object=lower.lm,scope=list(upper=formula(upper.lm),lower=~1),
                      direction=c('forward'), family = binomial());

# check results of forward
summary(forward.lm)

# create backward model
backward.lm <- stepAIC(object=upper.lm,direction=c('backward'), family = binomial());
summary(backward.lm)

# run models in both directions
stepwise.lm <- stepAIC(object=lower.lm,scope=list(upper=formula(upper.lm),lower=~1),
                       direction=c('both'), family = binomial());

# check model of both direction variable selection
summary(stepwise.lm)

cc_train$DEFAULT_forward <- predict(forward.lm, type = "response")
cc_train$DEFAULT_backward <- predict(backward.lm, type = "response")
cc_train$DEFAULT_both <- predict(stepwise.lm, type = "response")

AIC(forward.lm)
AIC(backward.lm)
AIC(stepwise.lm)
BIC(forward.lm)
BIC(backward.lm)
BIC(stepwise.lm)
print(-2*logLik(forward.lm, REML = TRUE))
print(-2*logLik(backward.lm, REML = TRUE))
print(-2*logLik(stepwise.lm, REML = TRUE))
mae(cc_train$DEFAULT, cc_train$DEFAULT_forward)
mae(cc_train$DEFAULT, cc_train$DEFAULT_backward)
mae(cc_train$DEFAULT, cc_train$DEFAULT_both)
ks_stat(actuals=cc_train$DEFAULT, predictedScores=cc_train$DEFAULT_forward)
ks_stat(actuals=cc_train$DEFAULT, predictedScores=cc_train$DEFAULT_backward)
ks_stat(actuals=cc_train$DEFAULT, predictedScores=cc_train$DEFAULT_both)

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
observations <- as.numeric(cc_train$DEFAULT)
predictions <- as.numeric(predict(backward.lm, type = "response"))
fun.auc(predictions, observations)

cc_train$DEFAULT_backward <- predict(backward.lm, type = "response")
summary(cc_train$DEFAULT)
summary(cc_train$DEFAULT_backward)

cc_train$DEFAULT_backward_ind <- ifelse(cc_train$DEFAULT_backward >= 0.5, 1,0)
summary(cc_train$DEFAULT_backward)
one <- cc_train %>%  filter(DEFAULT == 1)
zero <- cc_train %>%  filter(DEFAULT != 1)

# calculate the expected values
cc_test$DEFAULT_backward <- predict(backward.lm, newdata = cc_test, type = "response")
summary(cc_test$DEFAULT)
summary(cc_test$DEFAULT_backward)

# Run the function with the example data
observations <- as.numeric(cc_test$DEFAULT)
predictions <- as.numeric(predict(backward.lm, newdata = cc_test, type = "response"))

############################################################################### ROC

library(pROC)
g <- roc(DEFAULT ~ DEFAULT_logis, data = cc_train)
plot(g, col = "dark green")   

g <- roc(DEFAULT ~ DEFAULT_logis, data = cc_test)
plot(g, col = "dark green")  

lift1 <- lift(as.factor(DEFAULT) ~ DEFAULT_logis, data = cc_train)
lift1
xyplot(lift1, auto.key = list(columns = 20))
plotLift(DEFAULT_logis,as.factor(DEFAULT), data = cc_train)

require(ROCR)
pred <- prediction(cc_train$DEFAULT_logis,as.factor(cc_train$DEFAULT))
perf <- performance(pred,"tpr","fpr")
plot(perf, main="ROC curve", colorize=T)
abline()

# And then a lift chart
perf <- performance(pred,"lift","rpp")
plot(perf, main="lift curve", colorize=T)
perf

# write.csv(cc_train, file = "lift_data.csv")

############################################################################### KS Statistic

table(cc_train$DEFAULT)

decile.pts <- quantile(cc_train$DEFAULT_backward,
                       probs=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9));

cc_train$model.decile <- cut(cc_train$DEFAULT_backward,breaks=c(0,decile.pts,1),
                          labels=rev(c('01','02','03','04','05','06','07','08','09','10'))
);

head(cc_train)

# Note that we want the 'top decile' to be the highest model scores so we
# will reverse the labels.

# Check the min score in each model decile;
aggregate(cc_train$DEFAULT_backward,by=list(Decile=cc_train$model.decile),FUN=mean);

table(cc_train$model.decile)

cc_train$response <- ifelse(cc_train$DEFAULT_backward < .5,1,0)

table(cc_train$model.decile,cc_train$DEFAULT)

ks.table <- as.data.frame(list(Y0=table(cc_train$model.decile,cc_train$DEFAULT)[,1],
                               Y1=table(cc_train$model.decile,cc_train$DEFAULT)[,2],
                               Decile=rev(c('01','02','03','04','05','06','07','08','09','10'))
));


# Sort the data frame by decile;
ks.table[order(ks.table$Decile),]

############################################################################### KS Statistic TEST

# Now plug these values into your KS spreadsheet and compute the KS statistic;

table(cc_test$DEFAULT)

decile.pts <- quantile(cc_test$DEFAULT_backward,
                       probs=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9));

cc_test$model.decile <- cut(cc_test$DEFAULT_backward,breaks=c(0,decile.pts,1),
                             labels=rev(c('01','02','03','04','05','06','07','08','09','10'))
);

head(cc_test)

# Note that we want the 'top decile' to be the highest model scores so we
# will reverse the labels.

# Check the min score in each model decile;
aggregate(cc_test$DEFAULT_backward,by=list(Decile=cc_test$model.decile),FUN=mean);

table(cc_test$model.decile)

cc_test$response <- ifelse(cc_test$DEFAULT_backward < .5,1,0)

table(cc_test$model.decile,cc_test$DEFAULT)

ks.table <- as.data.frame(list(Y0=table(cc_test$model.decile,cc_test$DEFAULT)[,1],
                               Y1=table(cc_test$model.decile,cc_test$DEFAULT)[,2],
                               Decile=rev(c('01','02','03','04','05','06','07','08','09','10'))
));


# Sort the data frame by decile;
ks.table[order(ks.table$Decile),]


# Now plug these values into your KS spreadsheet and compute the KS statistic;