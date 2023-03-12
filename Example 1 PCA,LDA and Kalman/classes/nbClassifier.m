classdef nbClassifier < handle
    %NBCLASSIFIER NB Classifier
    %   Naive Bayesian classifier
    
    properties
        model
        pred_angle
    end
    
    methods
        function obj = nbClassifier()
            %NBCLASSIFIER Construct an instance of this class
            
        end
        
        function [fr_total, fr_avg, obj] = fr_features(obj,data,dt,N)
            %FR_FEATURES Calculates the firing rate of the data in bins of size dt.
            % data - given data struct
            % dt - time bin size
            % N - total number of samples length of
            % fr_total - spiking rate divided in bins
            % fr_avg - average spiking rate across bins

            [T,A] = size(data); %get trial and angle length

            acc = 1;
            fr_avg = zeros(T*A,98); % initialise variables
            fr_total = zeros(T*A,N/dt*98);
            for t=1:1:T
                for a=1:1:A
                    fr = zeros(98,length(0:dt:N)-1);
                    for u=1:1:98
                        var = data(t,a).spikes(u,1:N);
                        var(var==0) = NaN; % make zeros equal to NaN
                        count = histcounts([1:1:N].*var,0:dt:N); % count spikes in every dt bin until N
                        fr(u,:) = count/dt;
                    end
                    fr_avg(acc,:) = mean(fr,2); % get mean firing rate across bins
                    f = reshape(fr,size(fr,1)*size(fr,2),1);
                    fr_total(acc,:) = f; % get all firing rates ordered in 98 blocks of the same bin
                    acc = acc+1;
                end
            end
        end
   
        function [obj] = fit(obj,trainingData)
            %FIT(trainingData) Trains model based on training data
            
            [T,~] = size(trainingData); % get size of training data
    
            N = 560; % define end time

            [~,fr_avg] = obj.fr_features(trainingData,80,N); % obtaining firing rate feature space from training data
    
            % NB classifier training
            Y=repmat([1:1:8]',T,1); % generate labels for classifier 
            obj.model = fitcnb(fr_avg,Y,'DistributionNames','kernel'); % Naive Bayesian classifier
        end
        
        function [out,obj] = predict(obj,testData)
            %PREDICT(testData,N) uses trained model to generate labels on
            %test data
            
            N = length(testData.spikes);
            [~,fr_avg] = obj.fr_features(testData,80,N); % preprocess EEG data
            out = predict(obj.model,fr_avg); % classify angle from Naive Bayesian
            obj.pred_angle = out;
        end
    end
end