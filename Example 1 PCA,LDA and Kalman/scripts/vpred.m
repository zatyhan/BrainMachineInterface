%% Predicting initial velocity with PCR
clc; clear all; close all;
load monkeydata0;

% Set random number generator
rng(2013);
ix = randperm(length(trial));
% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

X_train = zeros(50*8, 98); 
Y_train = zeros(50*8, 2);
X_test = zeros(50*8, 98); 
Y_test = zeros(50*8, 2);
t_counter = 0;
for T = 1:length(trainingData)
    for A = 1:8
%         fprintf('Trial %d \n', t_counter);
        t_counter = t_counter + 1;
        % Training data
        Y_train(t_counter, :) = (trainingData(T, A).handPos(1:2, 320) - trainingData(T, A).handPos(1:2, 300))/20;
        X_train(t_counter, :) = mean( reshape(trainingData(T, A).spikes(:, 1:320), 98, 320), 2);
        % Testing data
        Y_test(t_counter, :) = (testData(T, A).handPos(1:2, 320) - testData(T, A).handPos(1:2, 300))/20;
        X_test(t_counter, :) = mean( reshape(testData(T, A).spikes(:, 1:320), 98, 320), 2);

    end
end

% Obtain optimal weights using PCR
n_pcr= 80;
X_train = [X_train, ones(size(X_train, 1), 1)];
X_test = [X_test, ones(size(X_test, 1), 1)];
[U,S,V] = svds(X_train, n_pcr);
W = (V*inv(S)*U')*Y_train;

% Compute test and train RMSE
rmse_train = sqrt(mean((Y_train - X_train * W).^2, 'all'));
rmse_test = sqrt(mean((Y_test - X_test * W).^2, 'all'));

% Visualizing vectors
Y_pred = X_test * W;

for i = 1:20
    figure(i);
%     fprintf('Trial %d' ,i)
    plotv([Y_test(i, 1), Y_pred(i, 1);
    Y_test(i, 2), Y_pred(i, 2)]);
end

%% Extracting the velocities of each trial
clc; clear all; close all;
load monkeydata0;

t_counter = 0;
V = zeros(800, 2, 35);
Acc = zeros(800, 2, 34);
for T = 1:100
    for A = 1:8
%         fprintf('Trial %d \n', t_counter);
        t_counter = t_counter + 1;
        timer = 1;
        for t = 320:20:length(trial(T, A).handPos)
            V(t_counter, :, timer) = (trial(T, A).handPos(1:2, t) - trial(T,A).handPos(1:2, t-20))/20;
            if t ~= 320
                Acc(t_counter, :, timer-1) = (V(t_counter, :, timer) - V(t_counter, :, timer-1))/20;
            end
            timer = timer + 1;
        end
        V(t_counter, 1, timer:end) = V(t_counter, 1, timer-1);
        V(t_counter, 2, timer:end) = V(t_counter, 2, timer-1);
        Acc(t_counter, 1, timer-1:end) = Acc(t_counter, 1, timer-2);
        Acc(t_counter, 2, timer-1:end) = Acc(t_counter, 2, timer-2);
        
    end
end

for i = 1:20
    figure(i); hold on;
    plot([320:20:320+33*20], squeeze(Acc(i, 1, :)));
    plot([320:20:320+33*20], squeeze(Acc(i, 2, :)));
    legend('X', 'Y');
    hold off;
    
end
