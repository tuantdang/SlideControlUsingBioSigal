clc, clear, close all;
load ionosphere;
tbl = array2table(X);
tbl.Y = Y;

rng('default') % For reproducibility
n = length(tbl.Y);
hpartition = cvpartition(n,'Holdout',0.3); % Nonstratified partition
idxTrain = training(hpartition);
tblTrain = tbl(idxTrain,:);
idxTest = test(hpartition);
tblTest = tbl(idxTest,:);

Mdl = fitcsvm(tblTrain, 'Y');
trainError = resubLoss(Mdl);
trainAccuracy = 1-trainError;

cvMdl = crossval(Mdl);
cvtrainError = kfoldLoss(cvMdl);
cvtrainAccuracy = 1-cvtrainError;

testError = loss(Mdl,tblTest,'Y');
testAccuracy = 1-testError;