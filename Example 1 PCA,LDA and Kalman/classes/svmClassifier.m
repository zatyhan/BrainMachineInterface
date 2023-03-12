classdef svmClassifier < handle
    %SVMCLASSIFIER SVM Classifier
    %   Support vector machine classifier
    
    properties
        model
        pred_angle
    end
    
    methods
        function obj = svmClassifier()
            %NBCLASSIFIER Construct an instance of this class
            
        end
        
        function [sim,obj] = gaussianKernel(obj, x1, x2, sigma)
            %RBFKERNEL returns a radial basis function kernel between x1 and x2
            %   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
            %   and returns the value in sim

            % Ensure that x1 and x2 are column vectors
            x1 = x1(:); x2 = x2(:);

            sim = exp(-(norm(x1 - x2) ^ 2) / (2 * (sigma ^ 2)));
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
        
        function [model,obj] = svmTrain(obj,X, Y, C, kernelFunction, ...
                            tol, max_passes)
            %SVMTRAIN Trains an SVM classifier using a simplified version of the SMO 
            %algorithm. 
            %   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
            %   SVM classifier and returns trained model. X is the matrix of training 
            %   examples.  Each row is a training example, and the jth column holds the 
            %   jth feature.  Y is a column matrix containing 1 for positive examples 
            %   and 0 for negative examples.  C is the standard SVM regularization 
            %   parameter.  tol is a tolerance value used for determining equality of 
            %   floating point numbers. max_passes controls the number of iterations
            %   over the dataset (without changes to alpha) before the algorithm quits.
            %
            % Note: This is a simplified version of the SMO algorithm for training
            %       SVMs. In practice, if you want to train an SVM classifier, we
            %       recommend using an optimized package such as:  
            %
            %           LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
            %           SVMLight (http://svmlight.joachims.org/)
            %
            %

            if ~exist('tol', 'var') || isempty(tol)
                tol = 1e-3;
            end

            if ~exist('max_passes', 'var') || isempty(max_passes)
                max_passes = 5;
            end

            % Data parameters
            m = size(X, 1);
            n = size(X, 2);

            % Map 0 to -1
            Y(Y==0) = -1;

            % Variables
            alphas = zeros(m, 1);
            b = 0;
            E = zeros(m, 1);
            passes = 0;
            eta = 0;
            L = 0;
            H = 0;

            % Pre-compute the Kernel Matrix since our dataset is small
            % (in practice, optimized SVM packages that handle large datasets
            %  gracefully will _not_ do this)
            % 
            % We have implemented optimized vectorized version of the Kernels here so
            % that the svm training will run faster.
            if strcmp(func2str(kernelFunction), 'linearKernel')
                % Vectorized computation for the Linear Kernel
                % This is equivalent to computing the kernel on every pair of examples
                K = X*X';
            elseif contains(func2str(kernelFunction), 'gaussianKernel')
                % Vectorized RBF Kernel
                % This is equivalent to computing the kernel on every pair of examples
                X2 = sum(X.^2, 2);
                K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
                K = kernelFunction(1, 0) .^ K;
            else
                % Pre-compute the Kernel Matrix
                % The following can be slow due to the lack of vectorization
                K = zeros(m);
                for i = 1:m
                    for j = i:m
                         K(i,j) = kernelFunction(X(i,:)', X(j,:)');
                         K(j,i) = K(i,j); %the matrix is symmetric
                    end
                end
            end

            % Train
            fprintf('\nTraining ...');
            dots = 12;
            while passes < max_passes

                num_changed_alphas = 0;
                for i = 1:m

                    % Calculate Ei = f(x(i)) - y(i) using (2). 
                    % E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
                    E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);

                    if ((Y(i)*E(i) < -tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0))

                        % In practice, there are many heuristics one can use to select
                        % the i and j. In this simplified code, we select them randomly.
                        j = ceil(m * rand());
                        while j == i  % Make sure i \neq j
                            j = ceil(m * rand());
                        end

                        % Calculate Ej = f(x(j)) - y(j) using (2).
                        E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);

                        % Save old alphas
                        alpha_i_old = alphas(i);
                        alpha_j_old = alphas(j);

                        % Compute L and H by (10) or (11). 
                        if (Y(i) == Y(j))
                            L = max(0, alphas(j) + alphas(i) - C);
                            H = min(C, alphas(j) + alphas(i));
                        else
                            L = max(0, alphas(j) - alphas(i));
                            H = min(C, C + alphas(j) - alphas(i));
                        end

                        if (L == H)
                            % continue to next i. 
                            continue;
                        end

                        % Compute eta by (14).
                        eta = 2 * K(i,j) - K(i,i) - K(j,j);
                        if (eta >= 0),
                            % continue to next i. 
                            continue;
                        end

                        % Compute and clip new value for alpha j using (12) and (15).
                        alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;

                        % Clip
                        alphas(j) = min (H, alphas(j));
                        alphas(j) = max (L, alphas(j));

                        % Check if change in alpha is significant
                        if (abs(alphas(j) - alpha_j_old) < tol)
                            % continue to next i. 
                            % replace anyway
                            alphas(j) = alpha_j_old;
                            continue;
                        end

                        % Determine value for alpha i using (16). 
                        alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));

                        % Compute b1 and b2 using (17) and (18) respectively. 
                        b1 = b - E(i) ...
                             - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                             - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
                        b2 = b - E(j) ...
                             - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                             - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

                        % Compute b by (19). 
                        if (0 < alphas(i) && alphas(i) < C)
                            b = b1;
                        elseif (0 < alphas(j) && alphas(j) < C)
                            b = b2;
                        else
                            b = (b1+b2)/2;
                        end

                        num_changed_alphas = num_changed_alphas + 1;

                    end

                end

                if (num_changed_alphas == 0)
                    passes = passes + 1;
                else
                    passes = 0;
                end

                fprintf('.');
                dots = dots + 1;
                if dots > 78
                    dots = 0;
                    fprintf('\n');
                end
                if exist('OCTAVE_VERSION')
                    fflush(stdout);
                end
            end
            fprintf(' Done! \n\n');

            % Save the model
            idx = alphas > 0;
            model.X= X(idx,:);
            model.y= Y(idx);
            model.kernelFunction = kernelFunction;
            model.b= b;
            model.alphas= alphas(idx);
            model.w = ((alphas.*Y)'*X)';

        end
        
        function [pred,obj] = svmPredict(obj,model, X)
            %SVMPREDICT returns a vector of predictions using a trained SVM model
            %(svmTrain). 
            %   pred = SVMPREDICT(model, X) returns a vector of predictions using a 
            %   trained SVM model (svmTrain). X is a mxn matrix where there each 
            %   example is a row. model is a svm model returned from svmTrain.
            %   predictions pred is a m x 1 column of predictions of {0, 1} values.
            %

            % Check if we are getting a column vector, if so, then assume that we only
            % need to do prediction for a single example
            if (size(X, 2) == 1)
                % Examples should be in rows
                X = X';
            end

            % Dataset 
            m = size(X, 1);
            p = zeros(m, 1);
            pred = zeros(m, 1);

            if strcmp(func2str(model.kernelFunction), 'linearKernel')
                % We can use the weights and bias directly if working with the 
                % linear kernel
                p = X * model.w + model.b;
            elseif contains(func2str(model.kernelFunction), 'gaussianKernel')
                % Vectorized RBF Kernel
                % This is equivalent to computing the kernel on every pair of examples
                X1 = sum(X.^2, 2);
                X2 = sum(model.X.^2, 2)';
                K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
                K = model.kernelFunction(1, 0) .^ K;
                K = bsxfun(@times, model.y', K);
                K = bsxfun(@times, model.alphas', K);
                p = sum(K, 2);
            else
                % Other Non-linear kernel
                for i = 1:m
                    prediction = 0;
                    for j = 1:size(model.X, 1)
                        prediction = prediction + ...
                            model.alphas(j) * model.y(j) * ...
                            model.kernelFunction(X(i,:)', model.X(j,:)');
                    end
                    p(i) = prediction + model.b;
                end
            end

            % Convert predictions into 0 / 1
            pred(p >= 0) =  1;
            pred(p <  0) =  0;

        end
   
        function [obj] = fit(obj,trainingData,C,s)
            %FIT(trainingData) Trains model based on training data
            % C - regularization (SVM parameter)
            % s - variance (SVM parameter)
            
            [T,~] = size(trainingData); % get size of training data
    
            N = 560; % define end time

            [~,fr_avg] = obj.fr_features(trainingData,80,N); % obtaining firing rate feature space from training data
            
            % SVM classifier training
            labels=repmat([1:1:8]',T,1); % generate labels for classifier 

            % CLASSIFICATION 1
            % right(0): 3, 4, 5, 6 - left(1): 1, 2, 7, 8
            l = or((labels<=2),(labels>=7));
            model1 = obj.svmTrain(fr_avg, double(l), C, @(x1, x2) obj.gaussianKernel(x1, x2, s));
            
            % CLASSIFICATION 2
            % 3, 4 (1) - 5, 6 (0)
            l2_low = labels(~l);
            l2_low_log = l2_low<5;
            idx = ~l;
            model2_low = obj.svmTrain(fr_avg(idx,:), double(l2_low_log), C, @(x1, x2) obj.gaussianKernel(x1, x2, s));
            
            % 1, 2 (1) - 7, 8 (0)
            l2_high = labels(l);
            l2_high_log = l2_high<3;
            idx = l;
            model2_high = obj.svmTrain(fr_avg(idx,:), double(l2_high_log), C, @(x1, x2) obj.gaussianKernel(x1, x2, s));
                
            % CLASSIFICATION 3
            % 1 (1) - 2 (0)
            l3_high_low = l2_high(l2_high_log); 
            l3_high_low_log = l3_high_low<2;
            idx = labels<3;
            model3_high_low = obj.svmTrain(fr_avg(idx,:), double(l3_high_low_log), C, @(x1, x2) obj.gaussianKernel(x1, x2, s));

            % 7 (1) - 8 (0)
            l3_high_high = l2_high(~l2_high_log); 
            l3_high_high_log = l3_high_high<8;
            idx = labels>6;
            model3_high_high = obj.svmTrain(fr_avg(idx,:), double(l3_high_high_log), C, @(x1, x2) obj.gaussianKernel(x1, x2, s));

            % 3 (1) - 4 (0)
            l3_low_low = l2_low(l2_low_log);
            l3_low_low_log = l3_low_low<4;
            idx = and((labels>2),(labels<5));
            model3_low_low = obj.svmTrain(fr_avg(idx,:), double(l3_low_low_log), C, @(x1, x2) obj.gaussianKernel(x1, x2, s));

            % 5 (1) - 6 (0)
            l3_low_high = l2_low(~l2_low_log); 
            l3_low_high_log = l3_low_high<6;
            idx = and((labels>4),(labels<7));
            model3_low_high = obj.svmTrain(fr_avg(idx,:), double(l3_low_high_log), C, @(x1, x2) obj.gaussianKernel(x1, x2, s));

            predict.model1_3456_1278 = model1;
            predict.model2_34_56 = model2_low;
            predict.model2_12_78 = model2_high;
            predict.model3_1_2 = model3_high_low;
            predict.model3_7_8 = model3_high_high;
            predict.model3_3_4 = model3_low_low;
            predict.model3_5_6 = model3_low_high;
            
            out = predict;
            obj.model = out;
        end
        
        function [out,obj] = predict(obj,testData)
            %PREDICT(testData,N) uses trained model to generate labels on
            %test data
            
            N = length(testData.spikes);
            [~,fr_avg] = obj.fr_features(testData,80,N); % preprocess EEG data

            pred_1 = obj.svmPredict(obj.model.model1_3456_1278,fr_avg(1,:));
            if pred_1 == 1 % left(1): 1, 2, 7, 8
                pred_2 = obj.svmPredict(obj.model.model2_12_78, fr_avg(1,:)); % 1, 2 (1) - 7, 8 (0)
                if pred_2 == 1 % 1 (1) - 2 (0)
                    pred_3 = obj.svmPredict(obj.model.model3_1_2, fr_avg(1,:));
                    if pred_3 == 1
                        obj.pred_angle = 1;
                    else 
                        obj.pred_angle = 2;
                    end
                elseif pred_2 == 0 % 7 (1) - 8 (0)
                    pred_3 = obj.svmPredict(obj.model.model3_7_8, fr_avg(1,:));
                    if pred_3 == 1 
                        obj.pred_angle = 7;
                    else
                        obj.pred_angle = 8;
                    end
                end
            elseif pred_1 == 0 % right(0): 3, 4, 5, 6 
                pred_2 = obj.svmPredict(obj.model.model2_34_56, fr_avg(1,:)); % 3, 4 (1) - 5, 6 (0)
                if pred_2 == 1 % 3 (1) - 4 (0)
                    pred_3 = obj.svmPredict(obj.model.model3_3_4, fr_avg(1,:));
                    if pred_3 == 1
                        obj.pred_angle = 3;
                    else 
                        obj.pred_angle = 4;
                    end
                elseif pred_2 == 0 % 5 (1) - 6 (0)
                    pred_3 = obj.svmPredict(obj.model.model3_5_6, fr_avg(1,:)); 
                    if pred_3 == 1 
                        obj.pred_angle = 5;
                    else
                        obj.pred_angle = 6;
                    end
                end
            end
            out = obj.pred_angle;
        end
    end
end