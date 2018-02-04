setwd('~/Documents/UIUC/SP18/CS498AML/HWs')
pima.data <- read.csv('pima-indians-diabetes.data', header = FALSE)
library(klaR)
library(caret)

# Data Preprocessing and select features 
label <- as.factor(pima.data[,9]) # select the last column as label
feat <- pima.data[, -c(9)] # select the colums except 9th as features

# Split the dataset to training set and test set 0.8: 0.2
train.idx <- createDataPartition(y=label, p=0.8, list=FALSE)
train.feat <- feat[train.idx, ]
train.label <- label[train.idx]
test.feat <- feat[-train.idx, ]
test.label <- label[-train.idx]

# Training model with SVM
svm <- svmlight(train.feat, train.label)
labels <- predict(svm, test.feat)
foo <- labels$class
sum(foo==test.label) / (sum(foo==test.label) + sum(!foo==test.label))
