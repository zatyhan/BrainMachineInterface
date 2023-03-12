function [x, y, newModelParameters] = positionEstimator(testData, modelParameters)
    % - test_data:
    % test_data(m).trialID
    % unique trial ID
    % test_data(m).startHandPos
    % 2x1 vector giving the [x y] position of the hand at the start
    % of the trial
    % test_data(m).decodedHandPos
    % [2xN] vector giving the hand position estimated by your
    % algorithm during the previous iterations. In this case, N is
    % the number of times your function has been called previously on
    % the same data sequence.
    % test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
    % in this case, t goes from 1 to the current time in steps of 20
    
    N = length(testData.spikes); % get trial length
    
    % Classification testing
    C_param = modelParameters.C_param; % extract classification parameters
    
    if N==320
        pred_angle = C_param.LDA1.predict(testData); % classify angle from LDA 
    elseif N==440
        pred_angle = C_param.LDA2.predict(testData);
    elseif N==560
        pred_angle = C_param.LDA3.predict(testData);
    else
        pred_angle = modelParameters.pred_angle;
    end 
    modelParameters.pred_angle = pred_angle;

    if N==560
        modelParameters.percentage = modelParameters.percentage+double(pred_angle==testData.angle);
        modelParameters.count = modelParameters.count+1;
    end
    
    % PCR regressor testing
    
    [x,y] = modelParameters.R_param.predict(testData,pred_angle,'avg');
    
    modelParameters.pred_pos = [x y];
    newModelParameters = modelParameters;
end

