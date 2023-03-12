%% BOOK RECURSIVE BAYESIAN FILTER

clearvars -except rate kin rate_train kin_train

addpath('funcs','data');

load('monkeydata_training.mat');

% MAP

dt = 20; % define time step
N = 560; % define trial time limit
range = 320:dt:N; % define relevant time points

[T,A] = size(trial); % get trial and angle size

for a = 1:1:A
    for bin = 1:length(range)
        numBin = range(bin)/dt; % get number of 20ms bins contained in the current N 
        [fr_total,~] = fr_features(trial,dt,N); % extract features from spikes data
        [x,y,~,~,~,~,~,~,~] = kinematics(trial); % extract kinematics from handPos data
        yTrain = squeeze(y(1:50,a,1:20:range(bin))-y(1:50,a,1)); % get trajectory at relevant locations
        yTrain = reshape(yTrain',0.5*T*numBin,1);
        yTest = squeeze(y(51:end,a,1:20:range(bin))-y(51:end,a,1)); % get trajectory at relevant locations
        yTest = reshape(yTest',0.5*T*numBin,1); 
        trial_idx = 1+T*(a-1);
        sTrain = squeeze(dt*fr_total(trial_idx:trial_idx+0.5*T-1,1:98*numBin)); % get neural data at relevant locations
        sTrain = transpose(reshape(sTrain',98,0.5*T*numBin));
        sTrain = [sTrain ones(0.5*T*numBin,1)];
        sTest = squeeze(dt*fr_total(trial_idx+0.5*T:a*T,1:98*numBin)); % create test data in the same format as train data
        sTest = transpose(reshape(sTest',98,0.5*T*numBin));
        sTest = [sTest ones(0.5*T*numBin,1)];
        
        yCenter = [min(yTrain):0.1:max(yTrain)]; % range of values y can take
        numNeuron = 98;
        for n = 1:numNeuron % fit Poisson distribution to relate spike rate/count with position
            coeff(n,:) = glmfit(yTrain,sTrain(:,n),'poisson');
            sFit(n,:) = exp(coeff(n,1) + coeff(n,2)*yCenter); 
        end

        for t = 1:size(sTest,1) % generate Poisson distribution for testing data based on training coefficients
            frTemp = sTest(t,1:numNeuron);
            prob = poisspdf(repmat(frTemp',1,length(yCenter)),sFit);
            prob(~any(prob,2),:) = []; % delete units with zero probability
            probSgivenY(t,:) = prod(prob); % assume independence between units
        end

        probY = histc(yTrain,linspace(min(yCenter),max(yCenter),length(yCenter))); % define probability of each relevant trajectory point based on training trajectory observations
        probY = probY(1:length(yCenter))/sum(probY(1:length(yCenter)));
        for t = 1:size(sTest,1) % infer P(Y|S) given P(S|Y) and P(Y)
            probYgivenS(t,:) = probSgivenY(t,:).*probY';
            [temp,maxInd] = max(probYgivenS(t,:)); % find maximum conditional probability P(Y|S)
            mapS(t) = yCenter(maxInd);
        end

        plot(1:size(sTest,1),yTest);
        hold on;
        plot(1:size(sTest,1),mapS);

        % Recursive Bayesian filter

        yDiff = diff(yTrain);
        yDiffEdge = [min(yDiff):0.01:max(yDiff)];
        yHist = histc(yDiff,yDiffEdge);
        probDiffY = yHist(1:length(yDiffEdge))/sum(yHist);
        figure;
        bar([min(yDiff):0.01:max(yDiff)],probDiffY); 

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
    end
end

