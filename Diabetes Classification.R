# libraries used

library(caret) #ML Model buidling package
library(tidyverse) #ggplot and dplyr
library(MASS) #Modern Applied Statistics with S
library(mlbench) #data sets from the UCI repository.
library(summarytools)
library(corrplot) #Correlation plot
library(gridExtra) #Multiple plot in single grip space
library(timeDate) 
library(pROC) #ROC
library(caTools) #AUC
library(rpart.plot) #CART Decision Tree
library(e1071) #imports graphics, grDevices, class, stats, methods, utils
library(graphics) #fourfoldplot

# Usually set.seed(123) for reproducibility. I am not doing it here due to software issue as I can not publish this document on its inclusion.
data(PimaIndiansDiabetes)
db <- PimaIndiansDiabetes
str(db)

summary(db$type)
summary(db)
str(db)
dim(db)


#Converted 0 to NA to input the mean of each variable for each missing data in the column 
db$glucose[db$glucose == 0] <- NA
db$pressure[db$pressure == 0] <- NA
db$triceps[db$triceps == 0] <- NA
db$insulin[db$insulin == 0] <- NA
db$mass[db$mass == 0] <- NA
db$age[db$age == 0] <- NA

str(db)
summary(db)


diabetes$Pregnancies [which(is.na(diabetes$Pregnancies))] = mean(diabetes$Pregnancies ,na.rm = TRUE)

#changing NA to the means
db$glucose [which(is.na(db$glucose))] = mean(db$glucose ,na.rm = TRUE)
db$pressure [which(is.na(db$pressure))] = mean(db$pressure ,na.rm = TRUE)
db$triceps [which(is.na(db$triceps))] = mean(db$triceps ,na.rm = TRUE)
db$insulin [which(is.na(db$insulin))] = mean(db$insulin ,na.rm = TRUE)
db$mass [which(is.na(db$mass))] = mean(db$mass ,na.rm = TRUE)
db$age [which(is.na(db$age))] = mean(db$age ,na.rm = TRUE)


#data exploration 
summary(db$type)
summary(db)
str(db)
dim(db)


#finding the means of each variable 
mean(db$pregnant)
mean(db$glucose)
mean(db$pressure)
mean(db$triceps)
mean(db$insulin)
mean(db$mass)
mean(db$pedigree)
mean(db$age)


#finding the sd of each variable 
sd(db$pregnant)
sd(db$glucose)
sd(db$pressure)
sd(db$triceps)
sd(db$triceps)
sd(db$mass)
sd(db$pedigree)
sd(db$age)

#Creating univariate boxplots

boxplot(db$pregnant,
        xlab = 'pregnant') #not normally distributed consider logging

boxplot(db$glucose,
        xlab = 'glucose')

boxplot(db$pressure,
        xlab = 'pressure')

boxplot(db$triceps,
        xlab = 'triceps')

boxplot(db$insulin,
        xlab = 'insulin') #not normally distributed consider logging

boxplot(db$mass,
        xlab = 'mass')

boxplot(db$pedigree,
        xlab = 'pedigree') #not normally distributed consider logging

boxplot(db$age,
        xlab = 'age') #not normally distributed consider logging


#logging variables that are not distributed normally 

#db$logpregnant = log(db$pregnant)
db$loginsulin = log(db$insulin)
db$logpedigree = log(db$pedigree)
db$logage = log(db$age)

summary(db$type)
summary(db)
str(db)
dim(db)

#removing the columns pregnant, insulin, pedigree, and age  that are not logged 

#db$pregnant = NULL
db$insulin = NULL
db$pedigree = NULL
db$age = NULL



summary(db$type)
summary(db)
str(db)
dim(db)



boxplot(db$logpregnant,
        xlab = 'logpregnant')

boxplot(db$loginsulin,
        xlab = 'loginsulin')

boxplot(db$logpedigree,
        xlab = 'logpedigree')

boxplot(db$logage,
        xlab = 'logage')



#store rows for partition
partition <- caret::createDataPartition(y = db$diabetes, times = 1, p = 0.7, list = FALSE)

# create training data set
train_set <- db[partition,]

# create testing data set, subtracting the rows partition to get remaining 30% of the data
test_set <- db[-partition,]

str(train_set)

str(test_set)

summary(train_set)

summarytools::descr(train_set)

ggplot(train_set, aes(train_set$diabetes, fill = diabetes)) + 
  geom_bar() +
  theme_bw() +
  labs(title = "Diabetes Classification", x = "Diabetes") +
  theme(plot.title = element_text(hjust = 0.5))

cor_data <- cor(train_set[,setdiff(names(train_set), 'diabetes')])
#Numerical Correlation Matrix
cor_data

# Correlation matrix plots
corrplot::corrplot(cor_data)

#####################################################################################################################################
# KNN - K Nearest Neighbours
#####################################################################################################################################
model_knn <- caret::train(diabetes ~., data = train_set,
                          method = "knn",
                          metric = "ROC",
                          tuneGrid = expand.grid(.k = c(3:10)),
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary))

model_knn

#final ROC value
model_knn$results[8,2]
model_knn$results


# prediction on Test data set
pred_knn <- predict(model_knn, test_set)
# Confusion Matrix 
cm_knn <- confusionMatrix(pred_knn, test_set$diabetes, positive="pos")

# Prediction Probabilities
pred_prob_knn <- predict(model_knn, test_set, type="prob")
# ROC value
roc_knn <- roc(test_set$diabetes, pred_prob_knn$pos)

# Confusion matrix 
cm_knn

# ROC Value for for KNN model
roc_knn

# AUC - Area under the curve
caTools::colAUC(pred_prob_knn$pos, test_set$diabetes, plotROC = T)



#####################################################################################################################################
# Random Forest Model
#####################################################################################################################################
model_forest <- caret::train(diabetes ~., data = train_set,
                             method = "ranger",
                             metric = "ROC",
                             trControl = trainControl(method = "cv", number = 10,
                                                      classProbs = T, summaryFunction = twoClassSummary))
model_forest

# final ROC Value
model_forest$results[6,4]
model_forest$results

# prediction on Test data set
pred_rf <- predict(model_forest, test_set)

# Confusion Matrix 
cm_rf <- confusionMatrix(pred_rf, test_set$diabetes, positive="pos")

# Prediction Probabilities
pred_prob_rf <- predict(model_forest, test_set, type="prob")
# ROC value
roc_rf <- roc(test_set$diabetes, pred_prob_rf$pos)

# Confusion Matrix for Random Forest Model
cm_rf

# ROC Value for Random Forest
roc_rf

# AUC - Area under the curve
caTools::colAUC(pred_prob_rf$pos, test_set$diabetes, plotROC = T)

#####################################################################################################################################
## naives bayes
#####################################################################################################################################
library(naivebayes)


model_nb <- caret::train(diabetes ~., data = train_set,
                          method = "naive_bayes",
                          metric = "ROC",
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary))


# prediction on Test data set
pred_nb <- predict(model_nb, test_set)

# Confusion Matrix 
cm_nb <- confusionMatrix(pred_nb, test_set$diabetes, positive="pos")

# Prediction Probabilities
pred_prob_nb <- predict(model_nb, test_set, type="prob")
# ROC value
roc_nb <- roc(test_set$diabetes, pred_prob_nb$pos)

# Confusion Matrix for Random Forest Model
cm_nb

# ROC Value for Random Forest
roc_nb

# AUC - Area under the curve
caTools::colAUC(pred_prob_nb$pos, test_set$diabetes, plotROC = T)






############################################################################################################
#Result comparison of the test set
############################################################################################################


result_rf <- c(cm_rf$byClass['Sensitivity'], cm_rf$byClass['Specificity'], cm_rf$byClass['Precision'], 
               cm_rf$byClass['Recall'], cm_rf$byClass['F1'], roc_rf$auc)

result_knn <- c(cm_knn$byClass['Sensitivity'], cm_knn$byClass['Specificity'], cm_knn$byClass['Precision'], 
                cm_knn$byClass['Recall'], cm_knn$byClass['F1'], roc_knn$auc)

result_nb <- c(cm_nb$byClass['Sensitivity'], cm_nb$byClass['Specificity'], cm_nb$byClass['Precision'], 
                cm_nb$byClass['Recall'], cm_nb$byClass['F1'], roc_nb$auc)


all_results <- data.frame(rbind(result_rf, result_knn, result_nb))
names(all_results) <- c("Sensitivity", "Specificity", "Precision", "Recall", "F1", "AUC")
all_results



########################################################################################################################
#ROC Value Comparison
########################################################################################################################


model_list <- list(Random_Forest = model_forest, KNN = model_knn, naive_bayes = model_nb)
resamples <- resamples(model_list)

#box plot
bwplot(resamples, metric="ROC")



























