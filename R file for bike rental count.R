############## bike rental count ##############

#clear all objects from R
rm(list = ls())

#set current working directory
setwd("C:/Users/nikhi/Desktop/Bike-Rental-Prediction-master")

#cross-check working directory
getwd()

#loading Libraries
x = c("ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "usdm", "unbalanced","dummies", "e1071",
      "DataCombine", "inTrees", "rpart","MASS", "stats")

#load Packages
lapply(x, require, character.only = TRUE)
rm(x)

#loading  dataset
bike_rental = read.csv("day.csv", header = T, na.strings = c(" ", "", "NA"))
str(bike_rental)
summary(bike_rental)
head(bike_rental,10)

#deleted atemp variable as it contributes similar info in temp variable (feature selection)

###############  Data pre-processing   #######################

############### 1.Exploratory data analysis #################

#Converting the variables to required data format
bike_rental$dteday = as.Date(as.character(bike_rental$dteday))

cols = c("season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit")
bike_rental[cols]= lapply(bike_rental[cols], factor)

str(bike_rental)

###############  2.Missing Value Analysis  ###############

#creating a dataframe with sum of missing values
missing_val = data.frame(apply(bike_rental,2,function(x){sum(is.na(x))})) 

#There is no missing value in the dataframe

##################### 3. Outlier Analysis  ##################

#1.temp variable
ggplot(data = bike_rental, aes(x = "", y = temp)) + geom_boxplot() 
# there is no outlier in temp variable

#2.hum variable
ggplot(data = bike_rental, aes(x = "", y = hum)) + geom_boxplot() 
#there is a negative outlier in hum variable

#3.windspeed variable
ggplot(data = bike_rental, aes(x = "", y = windspeed)) + geom_boxplot() 
#there is a positive outlier in windspeed variable

#4. casual variable
ggplot(data = bike_rental, aes(x = "", y = casual)) + geom_boxplot()
#there is a positive outlier in casual variable

#5. registered variable
ggplot(data = bike_rental, aes(x = "", y = registered)) + geom_boxplot()
#there is no outlier in registered variable

#Detect & Delete Outliers
cnames = c("hum", "windspeed", "casual")

for(i in cnames){
  print(i)
  val = bike_rental[,i][bike_rental[,i]%in% boxplot.stats(bike_rental[,i])$out]
  bike_rental = bike_rental[which(!bike_rental[,i]%in% val),]
  }

################   4. Feature selection   ##################

#Correlation analysis for continous variables
numeric_index = sapply(bike_rental,is.numeric) 
numeric_data = bike_rental[,numeric_index]

cnames = colnames(numeric_data)

corrgram(bike_rental[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")

#we cannot carryout chi-square test as target variable (cnt) is continous in nature

#Removing workingday because they dont contribute much to the independent variable
bike_rental = within(bike_rental, rm('instant', 'workingday'))

#ANOVA for categorical variables
aov_results = aov(cnt ~ season + yr + mnth + holiday + weekday + weathersit ,data = bike_rental)
summary(aov_results)

#Every variable has p-value less than 0.05 therefore we reject the null hypothesis.

######################## 5. Feature Scaling  ###################

#Normalisation for casual variable
print('casual')
bike_rental[,'casual'] = (bike_rental[,'casual'] - min(bike_rental[,'casual']))/
  (max(bike_rental[,'casual'] - min(bike_rental[,'casual'])))

#Normalisation for registered variable
print('registered')
bike_rental[,'registered'] = (bike_rental[,'registered'] - min(bike_rental[,'registered']))/
  (max(bike_rental[,'registered'] - min(bike_rental[,'registered'])))

head(bike_rental, 10)

################## machine learning algorithm ################

set.seed(123)
train.index = createDataPartition(bike_rental$cnt,p=0.75,list = FALSE)
train = bike_rental[train.index,]
test = bike_rental[-train.index,]

rmExcept(c("test","train", "bike_rental"))

train_feature = train[,c("season" ,"yr" ,"mnth","holiday","weekday","weathersit","temp","windspeed","casual","registered","cnt")]

test_feature = test[,c("season" ,"yr" ,"mnth","holiday","weekday","weathersit","temp","windspeed","casual","registered","cnt")]

###################  1. Linear regression   #################

#check multicollearity
library(usdm)
vif(train_feature[,-11])

warnings()

#check for normal distribution of data
qqnorm(train$cnt)
histogram(train$cnt)

#apply model on train
lm_model = lm(cnt ~., data = train_feature)
summary(lm_model)

#Predict on test
lm_predictions = predict(lm_model, test_feature[,-11])

#evalution of the model
#MAPE
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test_feature[,11],lm_predictions)

#MAPE for linear regression = 1.19541e-15
#Accuracy = 100 - 1.19541e-15

#Accuracy = 99.99%

####################  2.Decision Tree  #####################

#apply model on train
dt_model = rpart(cnt ~ ., data = train_feature, method = "anova")

#Predict for new test cases
dt_predictions = predict(dt_model, test_feature[,-11])

#evalution of the model
#MAPE
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test_feature[,11],dt_predictions)

#MAPE for decision tree = 11%
#Accuracy = 100 - 11

#Accuracy = 89%

################   3. Random forest   #####################

#apply model on train_data
rf_model = randomForest(cnt ~.,data=train_feature)

#Predict for new test cases
rf_predictions = predict(rf_model,test_feature[,-11])

#evalution of the model
#MAPE
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test_feature[,11],rf_predictions)

#MAPE for decision tree = 6.4%
#Accuracy = 100 - 6.4

#Accuracy = 93.6%

#conclusion: linear regression model is considered as best because it has  less % of error
