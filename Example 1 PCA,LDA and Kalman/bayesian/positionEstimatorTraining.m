function  [modelParameters] = positionEstimatorTraining(trainingData)
    % - trainingData:
    %     trainingData(n,k)              (n = trial id,  k = reaching angle)
    %     trainingData(n,k).trialId      unique number of the trial
    %     trainingData(n,k).spikes(i,t)  (i = neuron id, t = time)
    %     trainingData(n,k).handPos(d,t) (d = dimension [1-3], t = time)

    [T,A] = size(trainingData); % get size of training data
    
    C_param = struct;
    R_param = struct;
    N = 560; % define end time
    
    [~,fr_avg] = fr_features(trainingData,80,N); % obtaining firing rate feature space from training data
    
    % LDA classifier training
    
    Y=repmat([1:1:8]',T,1); % generate labels for classifier 
    C_param.Mdl_LDA = fitcdiscr(fr_avg,Y); % LDA classifier object
    
    % Maximum a posteriori (MAP) estimation training
    
    dt = 20; % define time step
    range = 320:dt:N; % define relevant time steps
    
    [fr_total,~] = fr_features(trainingData,20,N);
    [x,y,x_avg,y_avg,x_vel,y_vel,x_acc,y_acc,~] = kinematics(trainingData); % calculate x and y positions padded to maximum length
    
    for a = 1:A
        for u = 1:98
            idx_angle = (a-1)*T+1;
            fr_new = fr_total(idx_angle:idx_angle+T-1,u:98:end)';
            y_new = squeeze(y(:,a,range))';
            y_sampled(:,a) = reshape(y_new,T*length(range),1);
            coeff(a,u,:) = glmfit(y_sampled(:,a),reshape(fr_new,T*length(range),1),'poisson');
            f(a,u,:) = exp(coeff(a,u,1)+coeff(a,u,2)*y_avg(a,range));
        end
    end
    
    E_param.f = f;
    E_param.y = y_sampled;
    modelParameters.C_param = C_param;
    modelParameters.E_param = E_param;
    modelParameters.pred_angle = [];
end