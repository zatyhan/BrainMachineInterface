classdef ldaClassifier < handle
    %LDACLASSIFIER LDA Classifier
    %   Linear discriminant analysis classifier
    
    properties
        model
        opt
        pred_angle
        fr_norm
        P
    end
    
    methods
        function obj = ldaClassifier(opt)
            %LDACLASSIFIER Construct an instance of this class
            obj.opt = opt; % determines which type of lda to train
        end
        
        function [obj] = pca(obj,x,p)
            %PCA Calculates the principal components 
            % x - preprocessed firing rate in bins
            % p - number of components
            % P - principal components matrix
            
            C = cov(x);
            [V,D] = eig(C);
            [~,I] = maxk(abs(diag(D)),p);
            obj.P = V(:,I);
        end
        
        function [fr_avg, X, obj] = fr_features(obj,data,N)
            %FR_FEATURES Calculates the firing rate of the data in bins of size dt.
            % data - given data struct
            % dt - time bin size
            % N - total number of samples length of
            % fr_total - spiking rate divided in bins
            % fr_avg - average spiking rate across bins
            % X - average spiking rate across bins in different trial sections (prior to movement,
            % peri-movement and total)
            % Determines which 

            [T,A] = size(data); %get trial and angle length

            acc = 0;
            fr_avg = zeros(T*A,98); % initialise variables
            fr_avg1 = zeros(T*A,98); % initialise variables
            fr_avg2 = zeros(T*A,98); % initialise variables
            if obj.opt >= 2
                fr_avg3 = zeros(T*A,98); % initialise variables
            end
            if obj.opt == 3
                fr_avg4 = zeros(T*A,98);
            end
            for t=1:1:T
                for a=1:1:A
                    acc = acc+1;
                    fr_avg(acc,:) = mean(data(t,a).spikes(:,1:N),2); % get mean firing rate 
                    fr_avg1(acc,:) = mean(data(t,a).spikes(:,1:200),2); % get mean firing rate 
                    fr_avg2(acc,:) = mean(data(t,a).spikes(:,200:320),2); % get mean firing rate
                    if obj.opt >= 2
                        fr_avg3(acc,:) = mean(data(t,a).spikes(:,320:440),2); % get mean firing rate
                    end
                    if obj.opt == 3
                         fr_avg4(acc,:) = mean(data(t,a).spikes(:,440:560),2);
                    end
                end
            end
            if obj.opt == 2
                X = [fr_avg,fr_avg1,fr_avg2,fr_avg3];
            elseif obj.opt == 3
                X = [fr_avg,fr_avg1,fr_avg2, fr_avg3, fr_avg4];
            else
                X = [fr_avg,fr_avg1,fr_avg2];
            end
        end
   
        function [obj] = fit(obj,trainingData)
            %FIT(trainingData) Trains model based on training data
            
            [T,~] = size(trainingData); % get size of training data

            [~,X] = obj.fr_features(trainingData,560); % obtaining firing rate feature space from training data
            obj.fr_norm.mean = mean(X,1);
            obj.fr_norm.std = std(X,1);
            X = (X-obj.fr_norm.mean)./obj.fr_norm.std;
            X(isnan(X)) = 0;
            X(isinf(X)) = 0;
            obj.pca(X,15);
            X = X*obj.P;
            
            % LDA classifier training
            Y=repmat([1:1:8]',T,1); % generate labels for classifier 
            obj.model = fitcdiscr(X,Y); % LDA classifier object
        end
        
        function [out,obj] = predict(obj,testData)
            %PREDICT(testData,N) uses trained model to generate labels on
            %test data
            
            N = length(testData.spikes);
            [~,X] = obj.fr_features(testData,N); % preprocess EEG data
            X = (X-obj.fr_norm.mean)./obj.fr_norm.std;
            X(isnan(X)) = 0;
            X(isinf(X)) = 0;
            X = X*obj.P;
            out = predict(obj.model,X); % classify angle from LDA
            obj.pred_angle = out;
        end
    end
end