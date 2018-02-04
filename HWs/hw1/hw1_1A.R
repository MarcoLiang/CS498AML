setwd('~/Documents/UIUC/SP18/CS498AML/HWs')
pima.data <- read.csv('pima-indians-diabetes.data', header = FALSE)
library(klaR)
library(caret)

label <- pima.data[,9] # select the last column as label
feat <- pima.data[, -c(9)] # select the colums except 9th as features

# initial an array for recording the result of 10 iterations
train.score <- array(dim=10)
test.score <- array(dim=10)

# iterate 10 times for cross-validation
for (i in 1:10) {
  
  # Split the dataset to training set and test set 0.8:0.2
  train.idx <- createDataPartition(y=label, p=0.8, list=FALSE)
  
  train.feat <- feat[train.idx, ]
  train.label <- label[train.idx]
  test.feat <- feat[-train.idx, ]
  test.label <- label[-train.idx]
  
  # The index of the transctrion s.t. labeled as 1 (positive)
  train.idx.pos <- train.label > 0 # 1 -> pos || 0 -> negative
  train.feat.pos <- train.feat[train.idx.pos, ]
  train.feat.neg <- train.feat[!train.idx.pos, ]
  
  # Calculate mean and sd
  train.pos.mean <- sapply(train.feat.pos, mean, na.rm=TRUE)
  train.pos.sd <- sapply(train.feat.pos, sd, na.rm=TRUE)
  
  train.neg.mean <- sapply(train.feat.neg, mean, na.rm=TRUE)
  train.neg.sd <- sapply(train.feat.neg, sd, na.rm=TRUE)
  
  
  # Nomralize and Calculate the log likelihood on training set
  train.feat.pos.offset <- t(t(train.feat) - train.pos.mean)
  train.feat.pos.scaled <- t(t(train.feat.pos.offset) / train.pos.sd)
  train.feat.pos.prob <- -(1/2) * rowSums(apply(train.feat.pos.scaled, c(1, 2), 
                                                 function(x)x^2), na.rm=TRUE) - sum(log(train.pos.sd))
  train.feat.neg.offset <- t(t(train.feat) - train.neg.mean)
  train.feat.neg.scaled <- t(t(train.feat.neg.offset) / train.neg.sd)
  train.feat.neg.prob <- -(1/2) * rowSums(apply(train.feat.neg.scaled, c(1, 2), 
                                                  function(x)x^2), na.rm=TRUE) - sum(log(train.neg.sd))
  
  # Evaluate on training set
  pred.train <- train.feat.pos.prob > train.feat.neg.prob
  num.correct.train <- pred.train == train.label
  train.score[i] <- sum(num.correct.train) / (sum(num.correct.train) + sum(!num.correct.train))

 # Normalize and Calculate the log likelihood on test set
  test.feat.pos.offset <- t(t(test.feat) - train.pos.mean)
  test.feat.pos.scaled <- t(t(test.feat.pos.offset) / train.pos.sd)
  test.feat.pos.prob <- -(1/2) * rowSums(apply(test.feat.pos.scaled, c(1, 2), 
                                                function(x)x^2), na.rm=TRUE) - sum(log(train.pos.sd))
  test.feat.neg.offset <- t(t(test.feat) - train.neg.mean)
  test.feat.neg.scaled <- t(t(test.feat.neg.offset) / train.neg.sd)
  test.feat.neg.prob <- -(1/2) * rowSums(apply(test.feat.neg.scaled, c(1, 2), 
                                                function(x)x^2), na.rm=TRUE) - sum(log(train.neg.sd))
  
  # Evaluate o test set
  pred.test <- test.feat.pos.prob > test.feat.neg.prob
  num.correct.test <- pred.test == test.label
  test.score[i] <- sum(num.correct.test) / (sum(num.correct.test) + sum(!num.correct.test))
}
test.avg <- mean(test.score)
test.avg 

