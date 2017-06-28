# Loan_Prediction

---
title: "R Notebook"
output:
  github_document: default
  html_notebook: default
---

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BruFence and http://mlg.ulb.ac.be/ARTML
To try :
Over sampling
Under sampling
Smote
ROSE
Throw away minority examples and switch to an anomaly detection framework.
At the algorithm level, or after it:
Adjust the class weight (misclassification costs).
Adjust the decision threshold.
Modify an existing algorithm to be more sensitive to rare classes.
Construct an entirely new algorithm to perform well on imbalanced data.
HDDT
Tomek's Link
Ensemble

##Problem Statement

##Data Exploration

Lets load the dataset from csv file.
```{r,include=T,eval=F}
CCdata <- read.csv("creditcard.csv")
```

Lets see how our response variable is distributed.
```{r}
table(CCdata$Class)
```


```{r}
prop.table(table(CCdata$Class))
```

Thus we see that we have only 492 fraudaulent observations which account for only 0.173% of data.

```{r}
str(CCdata)
```
We see that besides the principal components of original features, we have other independent variables like Time and Amount.
Lets explore them further.


```{r}
library(ggplot2)

ggplot(subset(CCdata,CCdata$Class == 1),aes(x = Time)) + geom_bar(stat = "count",binwidth =3000,col = "black",fill = "white") + scale_x_continuous(breaks = seq(0,170000,10000)) + theme(axis.text.x = element_text(angle = 40))
```
We observe 2 spikes at 40,000 and 90,000 units of time. If information about the units was available we would be able to find hours of day when most frauds take place.


```{r}

ggplot(subset(CCdata,CCdata$Class == 1),aes(x = Amount)) + geom_histogram(col = "black",fill = "darkgreen") + scale_x_continuous(breaks = seq(0,2250,200))

```

We notice that most fraudulent transaction amount are less than 300.


Lets split the dataset into train and test set.
We will split the dataset so that the original ratio of Class variable is preserved.
```{r}

library(caTools)
set.seed(999)
index <- sample.split(CCdata$Class,SplitRatio = 70/100)
train <- CCdata[index,]
test <- CCdata[!index,]

train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)
```

##Data Modelling

###Model Evaluation Function
If we use accuracy to evaluate model performance it will mislead us since predicting all observations as not fraud will result in 99.8% accuracy. 
Thus we will use:- 
                  Confusion Matrix - to help us understand correct and incorrect predictions.
                  AUC - to 
```{r}
library(ROSE)
library(caret)
errormetrics <- function(x){
  list("Confusion Matrix" = confusionMatrix(test$Class,x)$table,"ROC" =  roc.curve(test$Class,x))
  }
```

###Training Base Models
Lets train 3 base models:- Logistic Regression,Decision Tree and SVM model.

```{r,include=T,eval=F}
LRmodel <- glm(Class~.,family = "binomial",data = train)
LRprobab <- predict(LRmodel,newdata = test,type = "response")
LRpred <- ifelse(pred >= 0.5,1,0)
```

Lets evaluate the performance of our base logistic regression model.

```{r}
errormetrics(LRpred)
```

We notice that most of the non fraud observations were correctly classified but about 2/3 fraudulent observations were incorrectly classified. 
Our priority should be to classify as much fraudulent observations as possible because classifying non fraud as fraud is better than classifying fraud as non fraud.

Lets see how Decision tree and SVM perform.

```{r,include=T,eval=F}
DTmodel <- rpart(Class~.,data = train,method = "class")
DTpred <- predict(DTmodel,newdata = test,type = "class")
```

```{r}
errormetrics(DTpred)
```
Decision tree performs better than logistic regression, classifying 116 fraudulent observations correctly but still misclassifies 32 observations.  





```{r,include=T,eval=F}
library(e1071)
SVMmodel <- svm(Class~.,data = train)
SVMpred <- predict(SVMmodel,newdata = test)
```

```{r}
errormetrics(SVMpred)

```


While SVM classifies 4 more non fraudulent observations than decision tree correctly,it provides no improvement in classifying fraudulent observations.


## Training Models on Oversampled Data
Oversampling replicates the observations from minority class,thus balancing the data.

```{r,include=T,eval=F}
trainOver <- ovun.sample(Class~.,data = train,method = "over", p = 0.5)
OStrain <- trainOver$data
```

```{r}
table(OStrain$Class)
```
Originally we had 199020 non fraud and 344 fraud observations in train set.Lets see distribution in oversampled train 
```{r}
table(OStrain$Class)

```
Thus we see that the distribution of Class is roughly equal.

Lets train models on oversampled train
```{r,include=T,eval=F}
LRmodelOS <- glm(Class~.,family = "binomial",data = OStrain)
LRprobOS <- predict(LRmodelOS,newdata = test,type = "response")
LRpredOS <-  ifelse(LRpredOS > 0.5,1,0)
```

```{r}
errormetrics(LRpredOS)
```
Thus the logistic regression model successfully classifies 138 observation giving a huge boost in performance compare to previous models.

```{r,include=T,eval=F}
DTmodelOS <- rpart(Class~.,data = OStrain,method = "class")
DTpredOS <- predict(DTmodelOS,newdata = test,type = "class")
```


```{r}
errormetrics(DTpredOS)
```

We observe that DT model provides no improvement over logistic regression model. 

```{r,include=T,eval=F}
SVMmodelOS <- svm(Class~.,OStrain,probability = TRUE)
SVMpredOS <- predict(SVMmodelOS,newdata = test)
```

```{r}
errormetrics(SVMpredOS)

```
SVM also does not perform better than logistic regression model.

## Training Models on Undersampled Data
Undersampling reduces the number of observations from majority class to make the data set balanced.
```{r,include=T,eval=F}
trainUnder <- ovun.sample(Class~.,data = train,method = "under", p = 0.5)
UStrain <- trainUnder$data
```
```{r}
table(UStrain$Class)

```
The non fraud observations has gone down to 342,giving us a balanced dataset.
Now lets train models on undersampled dataset.

```{r,include=T,eval=F}
LRmodelUS <- glm(Class~.,family = "binomial",data = UStrain)
 LRprobUS <- predict(LRmodelUS,newdata = test,type = "response")
 LRpredUS <-  ifelse(LRprobUS > 0.5,1,0)
```

```{r}
errormetrics(LRpredUS)
```


We notice that logistic regression model on undersampled train performs better than any of the previous models.


```{r,include=T,eval=F}
DTmodelUS <- rpart(Class~.,data = UStrain,method = "class")
DTpredUS <- predict(DTmodelUS,newdata = test,type = "class")
```



```{r}
errormetrics(DTpredUS)

```

Decision Tree on undersampled data performs much better than any of the models but misclassifies 5000 of non fraud observations.


```{r,include=T,eval=F}
SVMmodelUS <- svm(Class~.,UStrain,probability = TRUE)
SVMpredUS <- predict(SVMmodelUS,newdata = test)
```

```{r}
errormetrics(SVMpredUS)
```

SVM on undersampled data doesnt provide much improvement over previous models.


##Training models on ROSE dataset
ROSE (Random Over Sampling Examples) package helps us to generate artificial data based on sampling methods and smoothed bootstrap approach.
```{r,include=T,eval=F}

bootstraping - https://www.thoughtco.com/what-is-bootstrapping-in-statistics-3126172
```

```{r,include=T,eval=F}
trainR <- ROSE(Class~.,data = train)
ROSEtrain <- trainR$data
```

```{r}
table(ROSEtrain$Class)

```
Thus we observe that our dataset was minority class was oversampled by adding artificial data points and majority class was undersampled.

Now lets train models on this dataset.
```{r,include=T,eval=F}
LRmodelROSE <- glm(Class~.,family = "binomial",data = ROSEtrain)
LRprobROSE <- predict(LRmodelROSE,newdata = test,type = "response")
LRpredROSE <- ifelse(LRprobROSE > 0.5,1,0)
```

```{r}
errormetrics(LRpredROSE)
```
We see that logistic regression model does quite well in classifying both the classes but not good enough.
```{r,include=T,eval=F}
DTmodelROSE <- rpart(Class~.,data = ROSEtrain,method = "class")
DTpredROSE <- predict(DTmodelROSE,newdata = test,type = "class")
```

```{r}
errormetrics(DTpredROSE)
```
Decision Tree model provides no additional improvement over previous models.

```{r,include=T,eval=F}
SVMmodelROSE <- svm(Class~.,ROSEtrain,probability = TRUE)
SVMpredROSE <- predict(SVMmodelROSE,newdata = test)
```


```{r}
errormetrics(SVMpredROSE)
```
SVM does good job at classifying non fraud observations but misclassifies 19 fraud observations.

##Training models on SMOTE dataset

The Synthetic Minority Over-sampling TEchnique (SMOTE) is an oversampling approach that creates synthetic minority class samples.

```{r,include=T,eval=F}
library(DMwR)
SMOTEtrain <- SMOTE(Class ~.,data = train,perc.over = 200,k = 5,perc.under = 200)
```

```{r}
table(SMOTEtrain$Class)
```

As we can see from above, minority observations have increased to 1032.


Now lets train models on this dataset.
```{r,include=T,eval=F}
LRmodelSMOTE <- glm(Class~.,family = "binomial",data = SMOTEtrain)
LRprobSMOTE <- predict(LRmodelSMOTE,newdata = test,type = "response")
LRpredSMOTE <- ifelse(LRprobSMOTE > 0.5,1,0)
```

```{r}
errormetrics(LRpredSMOTE)

```


```{r,include=T,eval=F}
library(rpart)
DTmodelSMOTE <- rpart(Class~.,data = SMOTEtrain,method = "class")
DTpredSMOTE <- predict(DTmodelSMOTE,newdata = test,type = "class")
```

```{r}
errormetrics(DTpredSMOTE)
```

```{r,include=T,eval=F}
SVMmodelSMOTE <- svm(Class~.,SMOTEtrain,probability = TRUE)
SVMpredSMOTE <- predict(SVMmodelSMOTE,newdata = test)
```

```{r}
errormetrics(SVMpredSMOTE)

```


```{r,include=T,eval=F}
library(pROC)
roc(response =  test$Class,predictor =  as.numeric(SVMpredSMOTE),plot = T)
```

Class Weights
```{r}
model_weights <- ifelse(train$Class == 0,
nrow(train)/(2*table(train$Class)[1]),
nrow(train)/(2*table(train$Class)[2]))
```

```{r}
model_weights <- ifelse(train$Class == 0,
                        (1/table(train$Class)[1]) * 0.5,
                        (1/table(train$Class)[2]) * 0.5)

## MY weights (from py)
model_weights <- ifelse(train$Class == 0,
                         nrow(train)/(2*table(train$Class)[1]),
                         nrow(train)/(2*table(train$Class)[2]))

```


Change this
```{r,include=T,eval=F}
lmw <- glm(Class~.,data = train,family = binomial,weights = model_weights)
lmwprob <- predict(lmw,newdata = test,type = "response")
lmwpred <- ifelse(lmwprob > 0.5,1,0)
```

```{r}
errormetrics(lmwpred)
```

```{r,include=T,eval=F}
DTweighted <- rpart(Class ~., data = train,weights = model_weights,method = "class")
DTpredweighted <- predict(DTweighted,newdata = test,type = "class")
```

```{r}
errormetrics(DTpredweighted)
```


Change This.
```{r,include=T,eval=F}
SVMweighted2 <- svm(Class~.,data = train,class.weights = c("0" = 0.5,"1" = 289.77),probability = TRUE)
SVMweighted <- SVMweighted2
SVMweightedpred <- predict(SVMweighted,newdata = test)
```

```{r}
errormetrics(SVMweightedpred)

```
DT: type =  "prob"
LR: type = "response"



Cost Matrix
Anomaly Detection

Ensemble

```{r,include=T,eval=F}
LRprobUS <- predict(LRmodelUS,type = "response",newdata = test)

DTprobUS <- as.data.frame(predict(DTmodelUS,type = "prob",newdata = test))
SVMprobUS <- predict(SVMmodelUS,probability = T,newdata = test)
SVMprob <- as.data.frame(attr(SVMprobUS, "prob"))
SVMprobdummmy <- as.data.frame(SVMprob )


prob <- data.frame(LRprobUS,DTprobUS$`1`,SVMprobdummmy$`1`) #as.data.frame
```

```{r}
cor(prob)
```

```{r,include=T,eval=F}
WAprob <- (prob$LRprobUS*0.4) + (prob$DTprobUS..1.*0.3) + (prob$SVMprobdummmy..1.*0.3)
WApred <- ifelse(WAprob > 0.5,1,0)
```

```{r,include=T,eval=F}

predictions <- data.frame(LRpred,DTpred,SVMpred,LRpredOS,DTpredOS,SVMpredOS,LRpredUS,DTpredUS,SVMpredUS,LRpredROSE,DTpredROSE,SVMpredROSE,LRpredSMOTE,DTpredSMOTE,SVMpredSMOTE,lmwpred,DTpredweighted,SVMweightedpred,WApred)
```

```{r,include=T,eval=F}
auc <- data.frame()
for(i in 1:ncol(predictions)){
  a <- errormetrics(predictions[,i])
  #auc[i,1] <- colnames(predictions)[i]
  auc[i,1] <- colnames(predictions)[i]
  auc[i,2] <- a$ROC$auc
  
  for(j in 1:length(a$confumat$byClass)){
    auc[i,j+2] <- a$confumat$byClass[j]
  }
}

colnames(auc)[1] <- "PredictionName"
colnames(auc)[2] <- "AUC"
colnames(auc)[3:13] <- names(a$confumat$byClass)
```

```{r,include=T,eval=F}
probLR <- data.frame(LRprobSMOTE,LRprobOS,lmwprob)
```

```{R}
cor(probLR)
```

```{r,include=T,eval=F}
WAprobLR <- ((probLR$LRprobSMOTE * 0.4) + (probLR$LRprobOS * 0.3) + (probLR$lmwprob * 0.3))
WApred <- ifelse(WAprob > 0.5,1,0) 
```




```{r,include=T,eval=F}
probLR <- data.frame(LRprobSMOTE,DTprobUS$`1`,SVMprobdummmy$`1`)
```

```{r}
cor(probLR)
```

```{r,include=T,eval=F}
WAprob <- (probLR$LRprobSMOTE*0.4 + probLR$DTprobUS..1.*0.3 + probLR$SVMprobdummmy..1. * 0.3)

WApred <- ifelse(WAprob > 0.5,1,0)
```


Fraudulent transaction detector (positive class is "fraud"):
Optimize for sensitivity
FN as a variable
Because false positives (normal transactions that are flagged as possible fraud) are more acceptable than false negatives (fraudulent transactions that are not detected)



```{r,include=F,eval=F}
library(randomForest)

RFmodel250 <- randomForest(Class ~.,data = train,ntree = 250)
RFpred <- predict(RFmodel250,test)
errormetrics(RFpred)
```
Original KNN
combine train and test then scale
```{r,include=F,eval=F}
library(class)
Sctrain <- scale(train[,1:30])
Sctest <- scale(test[,1:30])

KNNpred <- knn(train = Sctrain,test = Sctest,cl = train$Class,k = 3)
errormetrics(KNNpred)
```

x
```{r,include=F,eval=F}

SctrainOS <- scale(OStrain[,1:30])
#Sctest <- scale(test[,1:30])

KNNpred <- knn(train = SctrainOS,test = Sctest,cl = OStrain$Class,k = 3)
errormetrics(KNNpred)


```
