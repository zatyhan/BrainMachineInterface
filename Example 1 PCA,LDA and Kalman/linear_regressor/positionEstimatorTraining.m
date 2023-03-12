function  [modelParameters] = positionEstimatorTraining(trainingData)
    % - trainingData:
    %     trainingData(t,a)              (t = trial id,  k = reaching angle)
    %     trainingData(t,a).trialId      unique number of the trial
    %     trainingData(t,a).spikes(u,n)  (i = neuron id, t = time)
    %     trainingData(t,a).handPos(p,n) (d = dimension [1-3], t = time)
    
    class = Classifier(); % create Classifier masterclass
    
    % Classification training
    C_param.LDA1 = class.LDA1.fit(trainingData);
    C_param.LDA2 = class.LDA2.fit(trainingData);
    C_param.LDA3 = class.LDA3.fit(trainingData);
    
    % PCR linear regressor training
    
    lin = Lin_regressor;
    
    R_param = lin.fit(trainingData);
    
    modelParameters.C_param = C_param;
    modelParameters.R_param = R_param;
    modelParameters.pred_angle = [];
    
end