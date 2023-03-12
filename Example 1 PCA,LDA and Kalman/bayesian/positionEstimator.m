function [x, y, newModelParameters] = positionEstimator(testData, modelParameters)
    % - test_data:
    %     test_data(m).trialID
    %         unique trial ID
    %     test_data(m).startHandPos
    %         2x1 vector giving the [x y] position of the hand at the start
    %         of the trial
    %     test_data(m).decodedHandPos
    %         [2xN] vector giving the hand position estimated by your
    %         algorithm during the previous iterations. In this case, N is
    %         the number of times your function has been called previously on
    %         the same data sequence.
    %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
    %     in this case, t goes from 1 to the current time in steps of 20

     % Classification testing
    
    N = length(testData.spikes); % get trial length
    
    if N==320 || N==400 || N==480 || N==560
        C_param = modelParameters.C_param; % extract LDA classification parameters
        [~,fr_avg] = fr_features(testData,80,N); % preprocess EEG data
        pred_angle = predict(C_param.Mdl_LDA,fr_avg); % classify angle from LDA 
        modelParameters.pred_angle = pred_angle;
    else
        pred_angle = modelParameters.pred_angle;
    end
    
    % Maximum a posteriori (MAP) estimator testing
    
    if N>560 % set N limit to 560
        N = 560;
    end
    dt = 20;
    range = [320:dt:560];
    
    [fr_total,~] = fr_features(testData,dt,N); % preprocess EEG data
    param = modelParameters.E_param; % get MAP estimator model parameters
    y = param.y;
    f = param.f;
    
    prob = poisspdf(repmat(fr_total,1,length(y)),f);
    
    
    idx_bin = length(fr_total)/98-(320/dt-1);
    update_x = param(pred_angle,idx_bin).update(:,1);
    update_y = param(pred_angle,idx_bin).update(:,2);
    update_x_vel = param(pred_angle,idx_bin).update(:,3);
    update_y_vel = param(pred_angle,idx_bin).update(:,4);
    update_x_acc = param(pred_angle,idx_bin).update(:,5);
    update_y_acc = param(pred_angle,idx_bin).update(:,6);
    fr_bin_avg = param(pred_angle,idx_bin).fr_bin_avg;
    x_avg_sampled = param(pred_angle,idx_bin).x_avg_sampled;
    y_avg_sampled = param(pred_angle,idx_bin).y_avg_sampled;
    x_vel_sampled = param(pred_angle,idx_bin).x_vel_sampled;
    y_vel_sampled = param(pred_angle,idx_bin).y_vel_sampled;
    x_acc_sampled = param(pred_angle,idx_bin).x_acc_sampled;
    y_acc_sampled = param(pred_angle,idx_bin).y_acc_sampled;
    
    x = (fr_total-fr_bin_avg)*update_x+x_avg_sampled;
    y = (fr_total-fr_bin_avg)*update_y+y_avg_sampled;

    vx = (fr_total-fr_bin_avg)*update_x_vel+x_vel_sampled;
    vy = (fr_total-fr_bin_avg)*update_y_vel+y_vel_sampled;

    acc_x = (fr_total-fr_bin_avg)*update_x_acc+x_acc_sampled;
    acc_y = (fr_total-fr_bin_avg)*update_y_acc+y_acc_sampled;
      
    x_prime = x_avg_sampled+vx;
    y_prime = y_avg_sampled+vy;
    vx_prime_2 = x_vel_sampled+acc_x;
    vy_prime_2 = y_vel_sampled+acc_y;
    x_prime_2 = x_avg_sampled+vx_prime_2;
    y_prime_2 = y_avg_sampled+vy_prime_2;
    
    x = (x+x_prime+x_prime_2)/3;
    y = (y+y_prime+y_prime_2)/3;

%     x = x_prime;
%     y = y_prime;
    
    modelParameters.pred_pos = [x y];
    newModelParameters = modelParameters;
end