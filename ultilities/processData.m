clc, clear, close all;
fileNames = [ ...
    "NormalTrain.csv" "NormalTest.csv" "GrindTrain.csv" ...
    "GrindTest.csv" "BlinkTrain.csv" "BlinkTest.csv" ...
    ];
nFiles = length(fileNames);
nClasses = 3;
chunkSize = 250;
nChunk = 10;

X = [];
Y = zeros(nChunk*nClasses*2, 1);
for i = 1:nFiles
    %disp(i);
    M = csvread(fileNames(i))';
    x = reshape(M, [chunkSize nChunk ]); % Change col -> Row
    X = [X; x'];
    if i <= 2
        Y(1:10*i) = 0;
    elseif i <= 4
        Y(21:10*i) = 1;
    elseif i <= 6
        Y(41:10*i) = 2;
    end
end
csvwrite("X.csv", X);
csvwrite("Y.csv", Y);
% Cross varidation (train: 70%, test: 30%)
%{
kFold = 5;
cv = cvpartition(size(X,1),'KFold', kFold);
for i = 1:kFold
    idx = cv.test(i);

    % Separate to training and test data
    XTrain = X(~idx,:);
    XTest  = X(idx,:);
    YTrain = Y(~idx);
    YTest = Y(idx);
    Mdl = fitcecoc(XTrain, YTrain); %SVN, KNN
    YPredict = predict(Mdl, XTest);
    YTestLabels = categorical(YTest);
    YPredictLabels = categorical(YPredict);
    
    trainError = resubLoss(Mdl);
    trainAccuracy = 1-trainError;
    
    testError = loss(Mdl, XTest, YTest);
    testAccuracy = 1-testError;
    fprintf("%f ,%f \n", trainAccuracy, testAccuracy);

    %figure;
    %plotconfusion(YTestLabels, YPredictLabels); 
end
%}