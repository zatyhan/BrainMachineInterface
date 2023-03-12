classdef Lin_regressor < handle
    %LIN_REGRESSOR Linear regressor decoder
    %   PCR linear regressor decoder
    
    properties
        model
        fr_total
        fr_avg
        x
        y
        x_avg
        y_avg
        x_vel
        y_vel
        x_acc
        y_acc
        l
    end
    
    methods
        function obj = Lin_regressor()
            %LIN_REGRESSOR Construct an instance of this class
            
        end
        
        function [P,obj] = PCA(obj,x,x_avg,p)
            %PCA Calculates the principal components 
            % x - preprocessed firing rate in bins
            % x_avg - trial average of preprocessed firing rate in bins
            % p - number of components
            % P - principal components matrix

            T = size(x,1);
            A=x'-x_avg';
            S=A'*A/T;
            [P,L]=eig(S);
            p=min(p,size(P,2));
            [~,ind]=maxk(diag(L),p);
            P=A*P(:,ind);
            P=P./sqrt(sum(P.^2));
        end
        
        function [obj] = fr_features(obj,data,dt,N)
            %FR_FEATURES Calculates the firing rate of the data in bins of size dt.
            % data - given data struct
            % dt - time bin size
            % N - total number of samples length of
            % fr_total - spiking rate divided in bins
            % fr_avg - average spiking rate across bins

            [T,A] = size(data); %get trial and angle length

            acc = 1;
            obj.fr_avg = zeros(T*A,98); % initialise variables
            obj.fr_total = zeros(T*A,N/dt*98);
            for t=1:1:T
                for a=1:1:A
                    fr = zeros(98,length(0:dt:N)-1);
                    for u=1:1:98
                        var = data(t,a).spikes(u,1:N);
                        var(var==0) = NaN; % make zeros equal to NaN
                        count = histcounts([1:1:N].*var,0:dt:N); % count spikes in every dt bin until N
                        fr(u,:) = count/dt;
                    end
                    obj.fr_avg(acc,:) = mean(fr,2); % get mean firing rate across bins
                    f = reshape(fr,size(fr,1)*size(fr,2),1);
                    obj.fr_total(acc,:) = f; % get all firing rates ordered in 98 blocks of the same bin
                    acc = acc+1;
                end
            end
        end
        
        function [obj] = kinematics(obj,data)
            % KINEMATICS Calculates multiple kinematic variables
            % data - given struct array
            % x - x position extended to maximum length (assuming stationarity) 
            % y - y position extended to maximum length (assuming stationarity) 
            % x_avg - mean x position extended to maximum length (assuming stationarity) 
            % y_avg - mean y position extended to maximum length (assuming stationarity) 
            % x_vel - x velocity extended to maximum length (assuming stationarity) 
            % y_vel - y velocity extended to maximum length (assuming stationarity)
            % x_acc - x acceleration extended to maximum length (assuming stationarity)
            % y_acc - y acceleration extended to maximum length (assuming stationarity)  
            % l - time length (N) of each trial

            data_cell = struct2cell(data); % convert to cell matrix
            max_length = @(x) length(x); % function handle to get max length of all elements in cell
            obj.l = cellfun(max_length,data_cell);
            obj.l = squeeze(obj.l(3,:,:)); % retain only handPos information

            L = max(obj.l,[],'all');

            [T,A] = size(data); % get dimensions of data
            obj.x_avg = zeros(A,L); % initialise variables
            obj.y_avg = zeros(A,L);
            obj.x = zeros(T,A,L);
            obj.y = zeros(T,A,L);
            obj.x_vel = zeros(T,A,L);
            obj.y_vel = zeros(T,A,L);
            obj.x_acc = zeros(T,A,L);
            obj.y_acc = zeros(T,A,L);
            for a = 1:1:A
                for t = 1:1:T
                    var_x = data(t,a).handPos(1,:);
                    var_y = data(t,a).handPos(2,:);
                    obj.x(t,a,:) = [var_x var_x(end)*ones(1,L-length(var_x))];
                    obj.y(t,a,:) = [var_y var_y(end)*ones(1,L-length(var_y))];
                    obj.x_vel(t,a,:) = [0 diff(squeeze(obj.x(t,a,:))')/0.02]; %calculate immediate velocity
                    obj.y_vel(t,a,:) = [0 diff(squeeze(obj.y(t,a,:))')/0.02];
                    obj.x_acc(t,a,:) = [0 0 diff(diff(squeeze(obj.x(t,a,:))')/0.02)/0.02]; %calculate immediate acceleration
                    obj.y_acc(t,a,:) = [0 0 diff(diff(squeeze(obj.y(t,a,:))')/0.02)/0.02];
                end
                obj.x_avg(a,:) = squeeze(mean(obj.x(:,a,:),1))';
                obj.y_avg(a,:) = squeeze(mean(obj.y(:,a,:),1))';
            end

        end
   
        function [obj] = fit(obj,trainingData)
            % PCR regressor training
            
            [T,A] = size(trainingData); % get size of training data
            dt = 20; % define time step
            N = 560; % define end time
            range = 320:dt:N; % define relevant time steps

            obj.fr_features(trainingData,dt,N);
            obj.kinematics(trainingData); % calculate kinematic variables padded to maximum length

            x_std = squeeze(std(obj.x,1)); % calculate standard deviation from the mean trajectory across trials for all angles
            y_std = squeeze(std(obj.y,1));

            x_detrended = zeros(T,A,size(obj.x,3));
            y_detrended = zeros(T,A,size(obj.x,3));
            for t = 1:T % Subtract position from mean position
                x_detrended(t,:,:) = squeeze(obj.x(t,:,:))-obj.x_avg;
                y_detrended(t,:,:) = squeeze(obj.y(t,:,:))-obj.y_avg;
            end

            x_detrended_sampled = x_detrended(:,:,range); % sample detrended data at relevant locations
            y_detrended_sampled = y_detrended(:,:,range);
            
            R_param = struct();

            for a = 1:A
                for bin = 1:length(range)
                    R_param(a,bin).x_avg_sampled = obj.x_avg(a,range(bin)); % store mean positions at relevant locations
                    R_param(a,bin).y_avg_sampled = obj.y_avg(a,range(bin));
                    R_param(a,bin).x_std_sampled = x_std(a,range(bin)); % store standard deviation of positions at relevant locations
                    R_param(a,bin).y_std_sampled = y_std(a,range(bin));
                    R_param(a,bin).x_vel_sampled = squeeze(mean(obj.x_vel(:,a,range(bin)),1)); % store mean velocity at relevant locations
                    R_param(a,bin).y_vel_sampled = squeeze(mean(obj.y_vel(:,a,range(bin)),1));
                    R_param(a,bin).x_acc_sampled = squeeze(mean(obj.x_acc(:,a,range(bin)),1)); % store mean acceleration at relevant locations
                    R_param(a,bin).y_acc_sampled = squeeze(mean(obj.y_acc(:,a,range(bin)),1));

                    idx_angle = (a-1)*T+1; % angle range index (for each a)
                    bin_idx = (range(bin)/dt); % bin range index (for each bin)
                    fr_bin = obj.fr_total(idx_angle:idx_angle+T-1,1:98*bin_idx);
                    bin_x = squeeze(x_detrended_sampled(:,a,bin));
                    bin_y = squeeze(y_detrended_sampled(:,a,bin));
                    bin_x_vel = squeeze(obj.x_vel(:,a,bin)-squeeze(mean(obj.x_vel(:,a,range(bin)),1))); % add detrended velocity
                    bin_y_vel = squeeze(obj.y_vel(:,a,bin)-squeeze(mean(obj.y_vel(:,a,range(bin)),1)));
                    bin_x_acc = squeeze(obj.x_acc(:,a,bin)-squeeze(mean(obj.x_acc(:,a,range(bin)),1))); % add detrended acceleration
                    bin_y_acc = squeeze(obj.y_acc(:,a,bin)-squeeze(mean(obj.y_acc(:,a,range(bin)),1)));
                    kin = [bin_x bin_y bin_x_vel bin_y_vel bin_x_acc bin_y_acc];

                    fr_bin_avg = mean(fr_bin,1);
                    R_param(a,bin).fr_bin_avg=fr_bin_avg; % store trial average firing rate per bin

                    % Use PCA to extract principle components
                    p = T-1;
                    P = obj.PCA(fr_bin,fr_bin_avg,p);
                    W = P'*(fr_bin'-fr_bin_avg');
                    update=P*(W*W')^(-1)*W*kin; % calculate linear regression
                    R_param(a,bin).update = update;
                end
            end
            obj.model = R_param;
        end
        
        function [x_pos,y_pos,obj] = predict(obj,testData,pred_angle,opt)
            %Decode testData with linear regressor model
            
            N = length(testData.spikes);
            if N>560 % set N limit to 560
                N = 560;
            end
            dt = 20;

            obj.fr_features(testData,dt,N); % preprocess EEG data

            idx_bin = length(obj.fr_total)/98-(320/dt-1);
            if strcmpi(opt,'pos') || strcmpi(opt,'posvel') || strcmpi(opt,'all')
                x_pred = (obj.fr_total-obj.model(pred_angle,idx_bin).fr_bin_avg)*obj.model(pred_angle,idx_bin).update(:,1)+obj.model(pred_angle,idx_bin).x_avg_sampled;
                y_pred = (obj.fr_total-obj.model(pred_angle,idx_bin).fr_bin_avg)*obj.model(pred_angle,idx_bin).update(:,2)+obj.model(pred_angle,idx_bin).y_avg_sampled;
            elseif strcmpi(opt,'vel') || strcmpi(opt,'posvel') || strcmpi(opt,'all') 
                vx = (obj.fr_total-obj.model(pred_angle,idx_bin).fr_bin_avg)*obj.model(pred_angle,idx_bin).update(:,3)+obj.model(pred_angle,idx_bin).x_vel_sampled;
                vy = (obj.fr_total-obj.model(pred_angle,idx_bin).fr_bin_avg)*obj.model(pred_angle,idx_bin).update(:,4)+obj.model(pred_angle,idx_bin).y_vel_sampled;
                x_prime = obj.model(pred_angle,idx_bin).x_avg_sampled+vx;
                y_prime = obj.model(pred_angle,idx_bin).y_avg_sampled+vy;
            elseif strcmpi(opt,'acc') || strcmpi(opt,'all')
                acc_x = (obj.fr_total-obj.model(pred_angle,idx_bin).fr_bin_avg)*obj.model(pred_angle,idx_bin).update(:,5)+obj.model(pred_angle,idx_bin).x_acc_sampled;
                acc_y = (obj.fr_total-obj.model(pred_angle,idx_bin).fr_bin_avg)*obj.model(pred_angle,idx_bin).update(:,6)+obj.model(pred_angle,idx_bin).y_acc_sampled;
                vx_prime_2 = obj.model(pred_angle,idx_bin).x_vel_sampled+acc_x;
                vy_prime_2 = obj.model(pred_angle,idx_bin).y_vel_sampled+acc_y;                
                x_prime_2 = obj.model(pred_angle,idx_bin).x_avg_sampled+vx_prime_2;
                y_prime_2 = obj.model(pred_angle,idx_bin).y_avg_sampled+vy_prime_2;    
            elseif strcmpi(opt,'avg')
                x_pos = obj.model(pred_angle,idx_bin).x_avg_sampled;
                y_pos = obj.model(pred_angle,idx_bin).y_avg_sampled;
            end

            if strcmpi(opt,'all')
                x_pos = (x_pred+x_prime+x_prime_2)/3;
                y_pos = (y_pred+y_prime+y_prime_2)/3;
            elseif strcmpi(opt,'posvel')
                x_pos = (x_pred+x_prime)/2;
                y_pos = (y_pred+y_prime)/2;
            elseif strcmpi(opt,'pos')
                x_pos = x_pred;
                y_pos = y_pred;
            elseif strcmpi(opt,'vel')
                x_pos = x_prime;
                y_pos = y_prime;
            elseif strcmpi(opt,'vel')
                x_pos = x_prime_2;
                y_pos = y_prime_2;
            end
        end
    end
end