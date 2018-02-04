#========== Setup Env ==========#
setwd('~/Documents/UIUC/SP18/CS498AML/HWs')

# Load Data
train <- read.csv("train.csv") 
test <- read.csv("test.csv")
train.feat <- train[, -c(1)]
train.label <- train[, 1]

#========== Visulization ==========#
##[the code showed on below was edied by source code
##from Dr. David Dalpiaz (dalpiaz2@illinois.edu)
##https://gist.github.com/daviddalpiaz/ae62ae5ccd0bada4b9acd6dbc9008706]
show_digit = function(dataset, idx, thold=5, col=gray(12:1/12), ...)
{
  dat <- matrix(unlist(train.feat[idx, ]), ncol = 28, byrow = T)
  mod = dat
  mod[mod[,] < thold] = 0
  image(t(mod)[,nrow(mod):1], col = col, axes=FALSE, ...)
  box(lty='solid', col='red', lwd=5)
}
#========== Preprocessing ==========#


#========== Test ==========#
show_digit(train.feat, 1)
lab <- train.label[1]
lab
