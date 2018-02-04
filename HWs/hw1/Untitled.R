setwd('~/Documents/UIUC/SP18/CS498AML/HWs')
pima.data <- read.csv('pima-indians-diabetes.data', header = FALSE)
library(klaR)
library(caret)


train.feat <- pima[,-c(9)]
train.leabel <- as.factor(wdat[,9])
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
trax<-bigx[wtd,]
tray<-bigy[wtd]
model<-train(trax, tray, 'nb', trControl=trainControl(method='cv', number=10))
teclasses<-predict(model,newdata=bigx[-wtd,])
confusionMatrix(data=teclasses, bigy[-wtd])

