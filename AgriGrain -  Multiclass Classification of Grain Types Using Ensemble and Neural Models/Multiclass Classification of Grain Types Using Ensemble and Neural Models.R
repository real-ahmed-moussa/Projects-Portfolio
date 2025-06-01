"
Ahmed Moussa
"

"================================== Clearing Memory and Loading Libraries ==================================="
rm(list=ls())
SEED = 123

library(ggplot2)
library(GGally)
library(caret)
library(randomForest)
library(gbm)
library(mclust)
library(nnet)
library(e1071)

"========================================== Loading the Dataset ============================================="
# Setting the Default Working Directory
setwd("path\\Data")

# Reading the Data
da <- read.csv("DB.csv")
da$Class <- as.factor(da$Class)
dim(da)

str(da)
head(da)
summary(da)
round(prop.table(table(da$Class)),2)

# Stratified Random Sampling
set.seed(SEED)
xda <- da[createDataPartition(da$Class, p = 0.1, list = FALSE),]
dim(xda)

str(xda)
head(xda)
summary(xda)
round(cor(xda[,-17]),2)
round(prop.table(table(xda$Class)),2)                                           # Stratified Sampling OK!

# Plots
ggplot(xda, aes(x=Class, y=Area,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=Perimeter,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=MajorAxisLength,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=MinorAxisLength,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=AspectRation,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=Eccentricity,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=ConvexArea,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=EquivDiameter,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=Extent,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=Solidity,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=roundness,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=Compactness,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=ShapeFactor1,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=ShapeFactor2,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=ShapeFactor3,color=Class)) + geom_boxplot()
ggplot(xda, aes(x=Class, y=ShapeFactor4,color=Class)) + geom_boxplot()

ggpairs(xda,aes(color = Class)) + theme_bw()

# Splitting the Dataset into Training and Testing Datasets (75%/25%)
ind_train <- createDataPartition(xda$Class, p = 0.75, list = FALSE)
ind_train <- as.vector(ind_train)

check_str <- xda[ind_train,]              
round(prop.table(table(check_str$Class)),2)                                     # Stratified Sampling OK!


"================================================ Bagging ==================================================="
# [1] Tuning the Model Parameters
# ===============================
set.seed(SEED)
bag.bean.tuned <- tune.randomForest(Class~.,data=xda[ind_train,],mtry=16,ntree=100*1:25,tunecontrol=tune.control(sampling="cross",cross=5))
summary(bag.bean.tuned)
plot(bag.bean.tuned$performances$error~bag.bean.tuned$performances$ntree,type="l",xlab="Number of Trees",ylab="Error",main="Bagging Error vs. Number of Trees")

# [2] Setting the Model
# =====================
set.seed(SEED)
bag.bean <- randomForest(Class~.,data=xda,subset=ind_train,mtry=16,ntree=2100,importance=TRUE,type="class")
bag.bean

# [3] Model Results on Training Dataset
# =====================================
bag.bean.misRate.training <- 1 - classAgreement(bag.bean$confusion)$diag        # Mis. Rate of Training Dataset
bag.bean.misRate.training
bag.bean.ARI.training <- classAgreement(bag.bean$confusion)$crand               # ARI of Training Dataset
bag.bean.ARI.training

# [4] Evaluating the Model on Testing Dataset
# ===========================================
bag.bean.predict.testing <- predict(bag.bean,xda[-ind_train,],type="class")
bag.bean.tab.testing <- table(xda[-ind_train,"Class"],bag.bean.predict.testing)
bag.bean.tab.testing

bag.bean.misRate.testing <- 1-classAgreement(bag.bean.tab.testing)$diag
bag.bean.misRate.testing
bag.bean.ARI.testing <- classAgreement(bag.bean.tab.testing)$crand
bag.bean.ARI.testing

importance(bag.bean)
varImpPlot(bag.bean)


"============================================= Random Forests ================================================"
# [1] Tuning the Model Parameters
# ===============================
set.seed(SEED)
rf.bean.tuned <- tune.randomForest(Class~.,data=xda[ind_train,],mtry=6:10,ntree=100*1:25,tunecontrol=tune.control(sampling="cross",cross=5))
summary(rf.bean.tuned)
plot(rf.bean.tuned)

# [2] Setting the Model
# =====================
set.seed(SEED)
rf.bean <- randomForest(Class~.,data=xda,subset=ind_train,mtry=6,ntree=500,importance=TRUE,type="class")
rf.bean

# [3] Model Results on Training Dataset
# =====================================
rf.bean.misRate.training <- 1 - classAgreement(rf.bean$confusion)$diag
rf.bean.misRate.training
rf.bean.ARI.training <- classAgreement(rf.bean$confusion)$diag
rf.bean.ARI.training

# [4] Evaluating the Model on Testing Dataset
# ===========================================
rf.bean.predict.testing <- predict(rf.bean,xda[-ind_train,],type="class")
rf.bean.tab.testing <- table(xda[-ind_train,"Class"],rf.bean.predict.testing)
rf.bean.tab.testing

rf.bean.misRate.testing <- 1-classAgreement(rf.bean.tab.testing)$diag
rf.bean.misRate.testing
rf.bean.ARI.testing <- classAgreement(rf.bean.tab.testing)$crand
rf.bean.ARI.testing

importance(rf.bean)
varImpPlot(rf.bean)


"================================================ Boosting ==================================================="
# [1] Tuning the Model Parameters
# ===============================
set.seed(SEED)
grid <- expand.grid(n.trees=c(200*1:10),interaction.depth=c(1:4),shrinkage=c(0.01,0.05,0.1),n.minobsinnode=c(20))
ctrl <- trainControl(method = "cv",number = 5)
unwantedoutput <- capture.output(GBMModel <- train(Class~.,data=xda[ind_train,],
                                                   method = "gbm", trControl = ctrl, tuneGrid = grid))
boost.bean.tuned <- GBMModel
print(boost.bean.tuned)
summary(boost.bean.tuned)
plot(boost.bean.tuned) 

# [2] Setting the Model
# =====================
set.seed(SEED)
boost.bean <- gbm(Class~.,data=xda[ind_train,],distribution="multinomial",n.trees=200,interaction.depth=3,shrinkage=0.05,n.minobsinnode=20)
boost.bean
summary(boost.bean)

# [3] Model Results on Training Dataset
# =====================================
boost.bean.predict.training <- predict(boost.bean,newdata=xda[ind_train,],distribution="multinomial",shrinkage=0.01,interaction.depth=2,type="response")
boost.class.pred.training <- rep(0,length(ind_train))
for(i in 1:length(ind_train)){
  which(boost.bean.predict.training[i,,1] == max(boost.bean.predict.training[i,,1])) -> boost.class.pred.training[i]
}
boost.bean.tab.training <- table(xda[ind_train,"Class"],boost.class.pred.training)
boost.bean.tab.training

boost.bean.misRate.training <- 1-classAgreement(boost.bean.tab.training)$diag
boost.bean.misRate.training
boost.bean.ARI.training <- classAgreement(boost.bean.tab.training)$crand
boost.bean.ARI.training

# [4] Evaluating the Model on Testing Dataset
# ===========================================
boost.bean.predict.testing <- predict(boost.bean,newdata=xda[-ind_train,],distribution="multinomial",shrinkage=0.05,interaction.depth=3,type="response")
boost.class.pred.testing <- rep(0,(nrow(xda)-length(ind_train)))
for(i in 1:(nrow(xda)-length(ind_train))){
  which(boost.bean.predict.testing[i,,1] == max(boost.bean.predict.testing[i,,1])) -> boost.class.pred.testing[i]
}
boost.bean.tab.testing <- table(xda[-ind_train,"Class"],boost.class.pred.testing)
boost.bean.tab.testing

boost.bean.misRate.testing <- 1-classAgreement(boost.bean.tab.testing)$diag
boost.bean.misRate.testing
boost.bean.ARI.testing <- classAgreement(boost.bean.tab.testing)$crand
boost.bean.ARI.testing


"====================================== Mixture Discriminant Analysis ========================================"
# [1] Setting the Model
# =====================
set.seed(SEED)
xda2 <- xda
xda2[,-17] <- scale(xda2[,-17])                                                 # Scaling the input variables.

beanMclustDA <- MclustDA(xda2[ind_train,-17],xda2[ind_train,17])
summy <- summary(beanMclustDA,newdata=xda2[-ind_train,-17],newclass=xda2[-ind_train,17])
summy

# [2] Model Results on Training Dataset
# =====================================
mda.bean.train.misRate.training <- 1-classAgreement(summy$tab)$diag 
mda.bean.train.misRate.training
mda.bean.train.ARI.training <- classAgreement(summy$tab)$crand
mda.bean.train.ARI.training

# [3] Evaluating the Model on Testing Dataset
# ===========================================
mda.bean.test.misRate.testing <- 1-classAgreement(summy$tab.newdata)$diag
mda.bean.test.misRate.testing
mda.bean.test.ARI.testing <- classAgreement(summy$tab.newdata)$crand
mda.bean.test.ARI.testing


"========================================= Artifical Neural Networks ========================================="
# [1] Tuning the Model Parameters
# ===============================
set.seed(SEED)
xda2 <- xda
xda2[,-17] <- scale(xda2[,-17])                                                 # Scaling the input variables.
nn.bean.tuned <- tune.nnet(Class~.,data=xda2[ind_train,],size=1:20,decay=0:5,tunecontrol=tune.control(sampling="cross",cross=5))
summary(nn.bean.tuned)
plot(nn.bean.tuned)

# [2] Setting the Model
# =====================
set.seed(SEED)
cls <- class.ind(xda$Class)

xda2 <- xda
xda2[,-17] <- scale(xda2[,-17])                                                 # Scaling the input variables.

nn_bean <- nnet(xda2[ind_train,-17],cls[ind_train,],size=11,decay=4,softmax=TRUE,maxit=500)

# [3] Model Results on Training Dataset
# =====================================
nn_pred.training <- predict(nn_bean,xda2[ind_train,-17],type="class")
nn.bean.tab.training <- table(xda2[ind_train,17],nn_pred.training)
nn.bean.tab.training

nn.bean.misRate.training <- 1-classAgreement(nn.bean.tab.training)$diag
nn.bean.misRate.training
nn.bean.ARI.training <- classAgreement(nn.bean.tab.training)$crand
nn.bean.ARI.training

# [4] Evaluating the Model on Testing Dataset
# ===========================================
nn_pred.testing <- predict(nn_bean,xda2[-ind_train,-17],type="class")
nn.bean.tab.testing <- table(xda2[-ind_train,17],nn_pred.testing)
nn.bean.tab.testing

nn.bean.misRate.testing <- 1-classAgreement(nn.bean.tab.testing)$diag
nn.bean.misRate.testing
nn.bean.ARI.testing <- classAgreement(nn.bean.tab.testing)$crand
nn.bean.ARI.testing

