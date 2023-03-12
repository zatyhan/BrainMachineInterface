classdef nnClassifier
    %NNCLASSIFIER neural network classifier
    %   Simple, shallow MLP classifier using pre-movement data
    
    properties
        model
        kernel
        options
        layers
    end
    
    methods
        function obj = nnClassifier()
            % Preprocessing information
            sigma = 0.025; 
            edges = [-3*sigma:.001:3*sigma];
            kernel = normpdf(edges,0,sigma); % Gaussian Kernel
            obj.kernel = kernel*.001;
            %Hyperparameters
            n_epochs = 150;
            lr = 2e-1;
            verb = false;
            % NN architechture
            obj.options = trainingOptions('sgdm', ...
                                        'MaxEpochs',n_epochs,...
                                        'InitialLearnRate',lr, ...
                                        'Verbose',verb, ...
                                        'LearnRateSchedule','piecewise', ...
                                        'LearnRateDropFactor',0.95, ...
                                        'LearnRateDropPeriod',5);
            obj.layers = [imageInputLayer([98*320, 1, 1], 'Normalization', 'zscore')
                        fullyConnectedLayer(100)
                        batchNormalizationLayer
                        reluLayer
                        dropoutLayer
                        fullyConnectedLayer(100)
                        batchNormalizationLayer
                        reluLayer
                        dropoutLayer
                        fullyConnectedLayer(100)
                        batchNormalizationLayer
                        reluLayer
                        dropoutLayer
                        fullyConnectedLayer(8)
                        softmaxLayer
                        classificationLayer];
        end
        function data = preprocess(obj, input_data)
            %PREPROCESS(input_data) Convolves spike data with gaussian
            %kernel
            
            [T, A] = size(input_data);
            data = zeros(98*320, 1, 1, T*A);
            t_counter = 0;
            for t = 1:T
                for a = 1:A
                    t_counter = t_counter + 1;
                    % Convoluted spike densities
                    freqs = zeros(98, 320);
                    for unit = 1:98
                        freqs(unit, :) = conv(input_data(t, a).spikes(unit, 1:320), obj.kernel, 'same');
                    end
                    % Input variable
                    data(:, 1, 1, t_counter) = reshape(freqs, 98*320, 1, 1, 1);
                end
            end
        end
        function obj = fit(obj, trainingData)
            %FIT(trainingData) Trains model based on training data
            
            % Data variables
            X = obj.preprocess(trainingData);
            Y = repmat([1:8], 1, length(trainingData));
            % Training model
            obj.model = trainNetwork(X, categorical(Y), obj.layers, obj.options);
        end
        function Y_pred = predict(obj, testData)
            %PREDICT(testData) uses trained model to generate labels on
            %test data
            
            N = length(testData);
            X_test = obj.preprocess(testData);
            Y_pred = predict(obj.model, X_test);
            [~,Y_pred] = max(Y_pred, [], 2);
        end
    end
end