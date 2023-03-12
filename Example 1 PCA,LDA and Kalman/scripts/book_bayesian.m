%% BOOK RECURSIVE BAYESIAN FILTER

clearvars -except rate kin rate_train kin_train

% MAP

numBin = length(kin_train);
yTrain = kin_train(1:numBin,:);
sTrain = rate_train(1:numBin,:);
sTrain = [sTrain ones(numBin,1)];
sTest = [rate(1:numBin,:) ones(numBin,1)];

yCenter = [0.5:1:14.5]; % range of values y can take
numNeuron = size(rate_train,2);
for n = 1:numNeuron 
    coeff(n,:) = glmfit(yTrain(:,2),sTrain(:,n),'poisson');
    sFit(n,:) = exp(coeff(n,1) + coeff(n,2)*yCenter); 
end

for t = 1:length(sTest) % loop over trials/bins
    frTemp = sTest(t,1:numNeuron);
    prob = poisspdf(repmat(frTemp',1,15),sFit);
    probSgivenY(t,:) = prod(prob);
end

probY = histc(yTrain(:,2),[0:15]);
probY = probY(1:15)/sum(probY(1:15));
for t = 1:length(sTest)
    probYgivenS(t,:) = probSgivenY(t,:).*probY';
    [temp,maxInd] = max(probYgivenS(t,:));
    mapS(t) = yCenter(maxInd);
end

plot(1:length(sTest),kin(:,2));
hold on;
plot(1:length(sTest),mapS);

% Recursive Bayesian filter

yDiff = diff(yTrain(:,2));
yDiffEdge = [-15.5:15.5];
yHist = histc(yDiff,yDiffEdge);
probDiffY = yHist(1:31)/sum(yHist);
figure;
bar([-15:15],probDiffY); 

probTemp = conv(probDiffY,ones(1,15)/15);
%Trim out the middle 
probPriorY(1,:) = probTemp(16:30)/sum(probTemp(16:30));

probPostY(1,:) = probPriorY(1,:).*probYgivenS(1,:);
probPostY(1,:) = probPostY(1,:)/sum(probPostY(1,:));
bayesY(1) = sum(probPostY(1,:).*yCenter);

%Recursive Bayesian decoder
for t = 2:length(sTest)
    %Convolve last estimate with error term for prior
    probTemp = conv(probPostY(t-1,:),probDiffY);
    probPriorY(t,:) = probTemp(16:30)/sum(probTemp(16:30)); %Combine prior with neural data for the posterior
    probPostY(t,:) = probPriorY(t,:).*probYgivenS(t,:); 
    probPostY(t,:) = probPostY(t,:)/sum(probPostY(t,:)); %Convert distribution to a single estimate of position 
    bayesY(t) = sum(probPostY(t,:).*yCenter);
end

figure;
plot(1:length(sTest),kin(:,2));
hold on;
plot(1:length(sTest),bayesY);

