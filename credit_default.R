# Lauren Camero
# 06.17.2019
 
############################################################################### LOAD PACKAGES
 
# install.packages("randomForest")
# install.packages("caret")
# install.packages("lattice")
# install.packages("gbm")
# install.packages("e1071")
# install.packages("jtools")
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
# install.packages("flux")
library(flux)
# install.packages("pROC")
library(pROC)
library(PRROC)

############################################################################### LOAD DATA

# set up the file path
my.path <- 'C:\\Users\\lcamero\\Downloads\\';
my.file <- paste(my.path,'credit_card_default.RData',sep='');


# Read the RData object using readRDS();
credit_card_default <- readRDS(my.file)
cc <- credit_card_default

############################################################################### EXPORE DATA

# examine the data
str(cc)
table(cc$data.group)
head(cc)
summary(cc)

# check descriptive statistics
table(cc$SEX)
table(cc$EDUCATION)
table(cc$MARRIAGE)
summary(cc$AGE)
table(cc$PAY_0)
table(cc$PAY_2)
table(cc$PAY_3)
table(cc$PAY_4)
table(cc$PAY_5)
table(cc$PAY_6)
table(cc$DEFAULT)
summary(cc$BILL_AMT1)
summary(cc$BILL_AMT2)
summary(cc$BILL_AMT3)
summary(cc$BILL_AMT4)
summary(cc$BILL_AMT4)
summary(cc$BILL_AMT6)
summary(cc$PAY_AMT1)
summary(cc$PAY_AMT2)
summary(cc$PAY_AMT3)
summary(cc$PAY_AMT4)
summary(cc$PAY_AMT5)
summary(cc$PAY_AMT6)
summary(cc$LIMIT_BAL)


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

# Bin age using woe.tree.binning
library(woeBinning)
library(OneR)

cc$target <- abs(as.numeric(cc$DEFAULT)-2);
age.tree <- woe.tree.binning(df=cc,target.var=c('target'),pred.var=c('AGE'))

# WOE plot for age bins;
woe.binning.plot(age.tree)
# Note that we got different bins;

# Score bins on data frame;
tree.df <- woe.binning.deploy(df=cc,binning=age.tree)
head(tree.df)
table(tree.df$age.in.years.binned)

# See the WOE Binning Table
woe.binning.table(age.tree)

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

############################################################################### EDA

# summary statistics of new variables
summary(cc$avg_bill_amt)
summary(cc$avg_pmt_amt)
summary(cc$pmt_ratio1)
summary(cc$pmt_ratio2)
summary(cc$pmt_ratio3)
summary(cc$pmt_ratio4)
summary(cc$pmt_ratio5)
summary(cc$avg_pmt_ratio)
summary(cc$max_bill_amt)
summary(cc$max_pmt_amt)
summary(cc$util)
summary(cc$util2)
summary(cc$util3)
summary(cc$util4)
summary(cc$util5)
summary(cc$util6)
summary(cc$avg_util)
summary(cc$bal_growth_6mo)
summary(cc$util_growth_6mo)
summary(cc$max_DLQ)
summary(cc$min_pmt_ratio)
summary(cc$education_by_age)

# use stargazer to 
# out.path <- 'C:\\Users\\lcamero\\Downloads\\';
# file.name <- 'summary_statistics.html';
# stargazer(cc, type=c('html'),out=paste(out.path,file.name,sep=''),
#           title=c('Summary Statistics'),
#           align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE, median=TRUE)


# EDA
# # recheck descriptive statistics of changed variables
# table(cc$EDUCATION)
# table(cc$MARRIAGE)
# table(cc$PAY_1)
# table(cc$PAY_2)
# table(cc$PAY_3)
# table(cc$PAY_4)
# table(cc$PAY_5)
# table(cc$PAY_6)
# table(cc$DEFAULT)

############################################################################### TRANSFORM VARIABLES

# review the boxplots 
par(mfrow=c(5,6))
boxplot(cc$LIMIT_BAL, main = "LIMIT_BAL")
boxplot(cc$BILL_AMT1, main = "BILL_AMT1")
boxplot(cc$BILL_AMT2, main = "BILL_AMT2")
boxplot(cc$BILL_AMT3, main = "BILL_AMT3")
boxplot(cc$BILL_AMT4, main = "BILL_AMT4")
boxplot(cc$BILL_AMT5, main = "BILL_AMT5")
boxplot(cc$BILL_AMT6, main = "BILL_AMT6")
boxplot(cc$PAY_AMT1, main = "PAY_AMT1")
boxplot(cc$PAY_AMT2, main = "PAY_AMT2")
boxplot(cc$PAY_AMT3, main = "PAY_AMT3")
boxplot(cc$PAY_AMT4, main = "PAY_AMT4")
boxplot(cc$PAY_AMT5, main = "PAY_AMT5")
boxplot(cc$PAY_AMT6, main = "PAY_AMT6")
boxplot(cc$avg_bill_amt, main = "Average Bill Amount")
boxplot(cc$avg_pmt_amt, main = "Average Payment Amount")
boxplot(cc$pmt_ratio1, main = "Payment Ratio")
boxplot(cc$max_bill_amt, main = "Max Bill Amount")
boxplot(cc$max_pmt_amt, main = "Max Payment Amount")
boxplot(cc$util, main = "Utilization")
boxplot(cc$avg_util, main = "Average Utilization")
boxplot(cc$bal_growth_6mo, main = "Balance Growth")
boxplot(cc$max_DLQ, main = "Maximum Delinquency")
boxplot(cc$avg_pmt_ratio, main = "Average Payment Ratio")
boxplot(cc$min_pmt_ratio, main = "Minimum Payment Ratio")
boxplot(cc$education_by_age, main = "Education x Age")
boxplot(cc$util_growth_6mo, main = "Utilization Growth")

# review the log of the boxplots
par(mfrow=c(5,5))
boxplot(log(cc$LIMIT_BAL), main = "LIMIT_BAL")
boxplot(log(cc$BILL_AMT1), main = "BILL_AMT1")
boxplot(log(cc$BILL_AMT2), main = "BILL_AMT2")
boxplot(log(cc$BILL_AMT3), main = "BILL_AMT3")
boxplot(log(cc$BILL_AMT4), main = "BILL_AMT4")
boxplot(log(cc$BILL_AMT5), main = "BILL_AMT5")
boxplot(log(cc$BILL_AMT6), main = "BILL_AMT6")
boxplot(log(cc$PAY_AMT1), main = "PAY_AMT1")
boxplot(log(cc$PAY_AMT2), main = "PAY_AMT2")
boxplot(log(cc$PAY_AMT3), main = "PAY_AMT3")
boxplot(log(cc$PAY_AMT4), main = "PAY_AMT4")
boxplot(log(cc$PAY_AMT5), main = "PAY_AMT5")
boxplot(log(cc$PAY_AMT6), main = "PAY_AMT6")
boxplot(log(cc$avg_bill_amt), main = "Average Bill Amount")
boxplot(log(cc$avg_pmt_amt), main = "Average Payment Amount")
boxplot(log(cc$pmt_ratio1), main = "Payment Ratio")
boxplot(log(cc$max_bill_amt), main = "Max Bill Amount")
boxplot(log(cc$max_pmt_amt), main = "Max Payment Amount")
boxplot(log(cc$util), main = "Utilization")
boxplot(log(cc$avg_util), main = "Average Utilization")
boxplot(log(cc$max_DLQ), main = "Maximum Delinquency")
boxplot(sqrt(cc$min_pmt_ratio), main = "Minimum Payment Ratio")
boxplot(log(cc$education_by_age), main = "Education x Age")


# transform the variables into the log
cc$LIMIT_BAL_log <- log(cc$LIMIT_BAL)
cc$AGE_log <- log(cc$AGE)
cc$BILL_AMT1_log <- log(cc$BILL_AMT1)
cc$BILL_AMT2_log <- log(cc$BILL_AMT2)
cc$BILL_AMT3_log <- log(cc$BILL_AMT3)
cc$BILL_AMT4_log <- log(cc$BILL_AMT4)
cc$BILL_AMT5_log <- log(cc$BILL_AMT5)
cc$BILL_AMT6_log <- log(cc$BILL_AMT6)
cc$PAY_AMT1_log <- log(cc$PAY_AMT1)
cc$PAY_AMT2_log <- log(cc$PAY_AMT2)
cc$PAY_AMT3_log <- log(cc$PAY_AMT3)
cc$PAY_AMT4_log <- log(cc$PAY_AMT4)
cc$PAY_AMT5_log <- log(cc$PAY_AMT5)
cc$PAY_AMT6_log <- log(cc$PAY_AMT6)
cc$avg_bill_amt_log <- log(cc$avg_bill_amt)
cc$avg_pmt_amt_log <- log(cc$avg_pmt_amt)
cc$pmt_ratio1_log <- log(cc$pmt_ratio1)
cc$max_bill_amt_log <- log(cc$max_bill_amt)
cc$max_pmt_amt_log <- log(cc$max_pmt_amt)
cc$util_log <- log(cc$util)
cc$avg_util_log <- log(cc$avg_util)
cc$max_DLQ_log <- log(cc$max_DLQ)
cc$min_pmt_ratio_sqrt <- sqrt(cc$min_pmt_ratio)
cc$education_by_age_log <- log(cc$education_by_age)

############################################################################### VISUALIZE VARIABLES

plot(cc$AGE~ cc$LIMIT_BAL)
p<-ggplot(cc, aes(x=LIMIT_BAL)) + 
  geom_histogram(color="black", fill="white")
p
p<-ggplot(cc, aes(x=BILL_AMT1)) + 
  geom_histogram(color="black", fill="white")
p
neg_bill <- subset(cc, cc$BILL_AMT1 <= -1)
p<-ggplot(neg_bill, aes(x=BILL_AMT1)) + 
  geom_histogram(color="black", fill="white")
p
p<-ggplot(cc, aes(x=SEX, y=BILL_AMT1)) +
  geom_boxplot()
p
summary(cc$LIMIT_BAL)
cc$balance_6 <- -cc$BILL_AMT6 + cc$PAY_AMT6
cc$balance_5 <- -cc$BILL_AMT5 + cc$PAY_AMT5 + cc$balance_6
cc$balance_4 <- -cc$BILL_AMT4 + cc$PAY_AMT4 + cc$balance_5
cc$balance_3 <- -cc$BILL_AMT3 + cc$PAY_AMT3 + cc$balance_4
cc$balance_2 <- -cc$BILL_AMT2 + cc$PAY_AMT2 + cc$balance_3
cc$balance_1 <- -cc$BILL_AMT1 + cc$PAY_AMT1 + cc$balance_2
cc$above_bal <- cc$LIMIT_BAL < cc$BILL_AMT1
sum(cc$train)
sum(cc$test)
sum(cc$validate)
cc$DEFAULT <- as.factor(cc$DEFAULT)

############################################################################### DISCRETE BINNING

# oneR binning for Limit Balance
options(scipen = 999)
bin.4 <- optbin(cc$DEFAULT ~ cc$LIMIT_BAL,method=c('logreg'));
table(bin.4)
aggregate(cc$DEFAULT, by=list(LIMIT_BAL=bin.4[,1]), FUN=mean)
cc$LIMiT_BAL_bin <- ifelse(cc$LIMIT_BAL >= 130000,1,0)

############################################################################### Correlation Matrix

cc_cor <- cor(cc[, (names(cc) %in% c("age_bin","avg_bill_amt","avg_pmt_amt","pmt_ratio1","avg_pmt_ratio"
                                     ,"max_bill_amt","max_pmt_amt","util","avg_util","bal_growth_6mo"
                                     ,"util_growth_6mo","max_DLQ","min_pmt_ratio","education_by_age"))])
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
# corrplot(cc_cor, method = "number", type = "upper", col = col(200), tl.col = "black", tl.srt = 45, diag = FALSE)

# use a scatter plot matrix to observe the correlations
cc_sp <- cc[, (names(cc) %in% c("DEFAULT", "avg_bill_amt","avg_pmt_amt","avg_pmt_ratio","avg_util","min_pmt_ratio"))]
pairs(cc_sp[,1:5],
      pch = 21,
      lower.panel = NULL,
      bg = c("orange", "dark grey")
      [unclass(cc_sp$DEFAULT)])

# Giant corrplot table
# par(mfrow=c(1,1))
# cc.cor <- cor(cc[c(2,6,7,13,14,15,16,17,18,19,20,21,22,23,24,25,33,34,35,40,41,42,43,49,50,51)])
# corrplot(cc.cor)
# names(cus) <- toupper(names(cus))
# cus_cor <- cor(cus[, !(names(cus) %in% c("FLT_ORIG_DT", "AIRPORT_PROCESS","INFLIGHT_CREW","AIRPORT_STAFF","INFLIGHT_SERVICE","OPD_FLT_IND"))])
# col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
# corrplot(cus_cor, method = "number", type = "upper", col = col(200), tl.col = "black", tl.srt = 45, diag = FALSE)
# out.path <- 'C:\\Users\\lcamero\\Downloads\\';
# file.name <- 'corr_matrix.html';
# cor.matrix <- cc.cor
# stargazer(cor.matrix, type=c('html'),out=paste(out.path,file.name,sep=''),
#           align=TRUE, digits=2, title='Correlation Matrix')

############################################################################### MODEL

# pick the model that is only the training data set 
cc_train <- subset(cc, train == 1)
cc_validate <- subset(cc, validate == 1)
cc_test <- subset(cc, test == 1)

############################################################################### RANDOM FOREST

# try a random forest
ranfor <- randomForest(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE 
                       + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6
                       + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5
                       + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4
                       + PAY_AMT5 + PAY_AMT6 + age_bin + avg_bill_amt + avg_pmt_amt + pmt_ratio1 
                       + pmt_ratio2 + pmt_ratio3 + pmt_ratio4 + pmt_ratio5 + avg_pmt_ratio 
                       + max_bill_amt + max_pmt_amt + util + util2 + util3 + util4 + util5 + util6 + avg_util
                       + bal_growth_6mo + util_growth_6mo + max_DLQ + min_pmt_ratio + education_by_age + LIMiT_BAL_bin
                       , data = cc_train, importance = TRUE, type = 'class')

# show model results
ranfor
summary(ranfor)
print(ranfor)
round(importance(ranfor),2)

# calculate the expected values
cc_train$DEFAULT_ranfor <- predict(ranfor, cc_train, type = 'class')
cc_train$DEFAULT_ranfor <- as.numeric(cc_train$DEFAULT_ranfor) - 1
summary(cc_train$DEFAULT_ranfor)
summary(cc_train$DEFAULT)

library(dplyr)
true <- cc_train %>%  filter(DEFAULT == 1)
false <- cc_train %>%  filter(DEFAULT != 1)
table(true$DEFAULT_ranfor)
table(false$DEFAULT_ranfor)

# calculate the expected values
cc_test$DEFAULT_ranfor <- predict(ranfor, cc_test, type = 'class')
cc_test$DEFAULT_ranfor <- as.numeric(cc_test$DEFAULT_ranfor) - 1
summary(cc_test$DEFAULT_ranfor)
summary(cc_test$DEFAULT)

true <- cc_test %>%  filter(DEFAULT == 1)
false <- cc_test %>%  filter(DEFAULT != 1)
table(true$DEFAULT_ranfor)
table(false$DEFAULT_ranfor)

# create a varibale importance plot
par(mfrow=c(1,1))
(VI_F=importance(ranfor))
varImp(ranfor)
varImpPlot(ranfor,type=2)
cc_train$DEFAULT_ranfor <- predict(ranfor, cc_train)
PRROC_obj <- roc.curve(scores.class0 = as.numeric(cc_train$DEFAULT), weights.class0=as.numeric(cc_train$DEFAULT_ranfor),
                       curve=TRUE)

# Build some example data
# Observed (truth) data as presence-absence (1-0)
# Predicted data as values ranging from 0 to 1

# Install the ROCR package
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
predictions <- as.numeric(predict(ranfor, cc_train))
fun.auc(predictions, observations)

# Run the function with the example data
observations <- as.numeric(cc_test$DEFAULT)
predictions <- as.numeric(predict(ranfor, cc_test))
fun.auc(predictions, observations)

############################################################################### GRADIENT BOOSTING

cc_train$DEFAULT <- as.numeric(cc_train$DEFAULT)

# run a gradient boosting model
boost = gbm(DEFAULT ~ SEX + EDUCATION + MARRIAGE + PAY_1 + PAY_3 + PAY_4 + PAY_5 
            + PAY_6 + BILL_AMT4 + PAY_AMT1 + PAY_AMT4 + PAY_AMT5 + age_bin 
            + avg_bill_amt + avg_pmt_amt + pmt_ratio2 + pmt_ratio3 + pmt_ratio4 
            + pmt_ratio5 + avg_pmt_ratio + max_bill_amt + max_pmt_amt + util 
            + avg_util + bal_growth_6mo + util_growth_6mo + max_DLQ 
            + min_pmt_ratio + education_by_age + LIMiT_BAL_bin
            ,data = cc_train, n.trees = 100
            , shrinkage = 0.01, interaction.depth = 4)

boost = gbm(DEFAULT ~ avg_bill_amt ,data = cc_train, n.trees = 100
            , shrinkage = 0.01)

cc_short <- cc_train[, (names(cc_train) %in% c("DEFAULT","age_bin","avg_bill_amt","avg_pmt_amt","pmt_ratio1","avg_pmt_ratio"
                                               ,"max_bill_amt","max_pmt_amt","util","avg_util","bal_growth_6mo"
                                               ,"util_growth_6mo","max_DLQ","min_pmt_ratio","education_by_age"))]
cc_train$DEFAULT <- as.numeric(cc_train$DEFAULT)
# for reproducibility
set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = DEFAULT ~ .,
  distribution = "bernoulli",
  data = cc_short,
  n.trees = 100,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 3,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  



# print results
print(gbm.fit)

?gbm

# show model results
boost
summary(boost)

cc_train$DEFAULT <- as.factor(cc_train$DEFAULT)

# calculate the expected values
cc_train$DEFAULT_boost <- predict(boost, cc_train, distribution = "bernoulli")
cc_train$DEFAULT_boost <- predict(boost, cc_test, n.trees = 10000)
summary(cc_train$DEFAULT)
summary(cc_train$DEFAULT_boost)


true <- cc_train %>%  filter(DEFAULT == 1)
false <- cc_train %>%  filter(DEFAULT != 1)
table(true$DEFAULT_boost)
table(false$DEFAULT_boost)

# calculate the expected values
cc_test$DEFAULT_ranfor <- predict(ranfor, cc_test, type = 'class')
cc_test$DEFAULT_ranfor <- as.numeric(cc_test$DEFAULT_ranfor) - 1
summary(cc_test$DEFAULT_ranfor)
summary(cc_test$DEFAULT)

true <- cc_test %>%  filter(DEFAULT == 1)
false <- cc_test %>%  filter(DEFAULT != 1)
table(true$DEFAULT_ranfor)
table(false$DEFAULT_ranfor)

# Run the function with the example data
observations <- as.numeric(cc_train$DEFAULT)
predictions <- as.numeric(predict(ranfor, cc_train))
fun.auc(predictions, observations)

# Run the function with the example data
observations <- as.numeric(cc_test$DEFAULT)
predictions <- as.numeric(predict(ranfor, cc_test))
fun.auc(predictions, observations)

############################################################################### LOGISTIC REGRESSION

# run logistic regression
logis <- glm(DEFAULT ~ PAY_1 + max_bill_amt + max_DLQ + util + avg_pmt_amt
             + age_bin + education_by_age, data = cc_train, family = binomial())

# check logistic regression results of the full model
summary(logis)

# use variable selection method: stepwise
# Define the upper model as the FULL model
upper.lm <- glm(DEFAULT ~ PAY_1 + max_bill_amt + max_DLQ + util + avg_pmt_amt
                + age_bin + education_by_age, data = cc_train, family = binomial());

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

AIC(forward.lm)
AIC(backward.lm)
AIC(stepwise.lm)
BIC(forward.lm)

# calculate theforward.lm expected values
cc_train$xdefault2 <- as.factor(round(predict.glm(stepwise.lm, cc_train, type = "response")))
cc_test$xdefault2 <- round(round(predict(stepwise.lm, cc_test, type = "response")))
cc_validate$xdefault2 <- round(round(predict(stepwise.lm, cc_validate, type = "response")))

# check the actual default 
table(cc_train$DEFAULT)
table(cc_train$xdefault2)
table(cc_test$DEFAULT)
table(cc_test$xdefault2)
table(cc_validate$DEFAULT)
table(cc_validate$xdefault2)

############################################################################### SVM

# run an svm model
svm = svm(DEFAULT ~ PAY_1 + max_bill_amt + max_DLQ + util + avg_pmt_amt
          + age_bin + education_by_age
          , data = cc_train, kernel = "linear", cost = 10, scale = FALSE)

?svm
# show svm model
print(svm)

# plot the svm results
plot(svm, data = cc_train, education_by_age ~ max_DLQ)

# calculate the expected values
svm_train =  predict(svm, cc_train)
svm_test =  predict(svm, cc_test)

# calculate the expected values
cc_train$DEFAULT_svm <- predict(svm, cc_train)
cc_test$DEFAULT_svm <- predict(svm, cc_test)

library(dplyr)
one <- cc_train %>%  filter(DEFAULT == 1)
zero <- cc_train %>%  filter(DEFAULT != 1)
table(one$DEFAULT_ranfor)
table(zero$DEFAULT_ranfor)

# calculate the expected values
cc_test$DEFAULT_ranfor <- predict(ranfor, cc_test, type = 'class')
cc_test$DEFAULT_ranfor <- as.numeric(cc_test$DEFAULT_ranfor) - 1
summary(cc_test$DEFAULT_ranfor)
summary(cc_test$DEFAULT)

true <- cc_test %>%  filter(DEFAULT == 1)
false <- cc_test %>%  filter(DEFAULT != 1)
table(true$DEFAULT_ranfor)
table(false$DEFAULT_ranfor)

# Run the function with the example data
observations <- as.numeric(cc_train$DEFAULT)
predictions <- as.numeric(svm_test)
fun.auc(predictions, observations)

# Run the function with the example data
observations <- as.numeric(cc_test$DEFAULT)
predictions <- as.numeric(predict(ranfor, cc_test))
fun.auc(predictions, observations)

############################################################################### NAIVE BAYES

library(naivebayes)
nb <- naive_bayes(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE 
                    + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6
                    + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5
                    + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4
                    + PAY_AMT5 + PAY_AMT6 + age_bin + avg_bill_amt + pmt_ratio1 
                    + pmt_ratio2 + pmt_ratio3 + pmt_ratio4 + pmt_ratio5 + avg_pmt_ratio 
                    + max_bill_amt + util + util2 + util3 + util4 + util5 + util6 + avg_util
                    + bal_growth_6mo + util_growth_6mo + max_DLQ + min_pmt_ratio + education_by_age 
                    ,data = cc_train)

# Compare the table to the table above;
nb

# What else do we get?
summary(nb)
names(nb)

nb$levels
nb$laplace
nb$data

# Plot Naive Bayes probabilities;
# Note that this is a degenerate plotting option since there is only one predictor;
plot(nb)

# Predict the class;
predicted.class <- predict(nb);

# calculate the expected values
cc_train$DEFAULT_nb <- predict(nb,cc_train)
cc_test$DEFAULT_nb <- predict(nb, cc_test)
summary(cc_train$DEFAULT_nb)
summary(cc_train$DEFAULT)

library(dplyr)
one <- cc_train %>%  filter(DEFAULT == "1")
zero <- cc_train %>%  filter(DEFAULT != "1")
table(one$DEFAULT_nb)
table(zero$DEFAULT_nb)

one <- cc_test %>%  filter(DEFAULT == "1")
zero <- cc_test %>%  filter(DEFAULT != "1")
table(one$DEFAULT_nb)
table(zero$DEFAULT_nb)

# Run the function with the example data
observations <- as.numeric(cc_train$DEFAULT)
predictions <- as.numeric(predict(nb, cc_train))
fun.auc(predictions, observations)

# Run the function with the example data
observations <- as.numeric(cc_test$DEFAULT)
predictions <- as.numeric(predict(nb, cc_test))
fun.auc(predictions, observations)

############################################################################### NAIVE BAYES TRY 2

# naive model
predictor.df <- cc_train[, (names(cc_train) %in% c("PAY_1", "MAX_BILL_AMT","MAX_DLQ","UTIL","AVG_PMT_AMT"
                                                   ,"age_bin","education_by_age"))]
nb.2 <- naive_bayes(x=predictor.df,y=cc_train$DEFAULT)


# Look at output;
summary(nb.2)
plot(nb.2)

# Plot Naive Bayes probabilities;
plot(nb.2, which=c('PAY_1'))

# Open additional graphics window;
X11()
plot(nb.2, which=c("MAX_BILL_AMT"))


# Predict the class;
predicted.class <- predict(nb.2);
mean(predicted.class=="DEFAULT")
xdefault5_train = predict(nb.2, cc_train)
xdefault5_test = predict(nb.2, cc_test)
