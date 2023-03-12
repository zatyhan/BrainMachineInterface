%% Gaussian model for angle encoding
set(groot, 'defaultTextInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
clc; clear variables; close all;

load('monkeydata_training.mat');

data = extractfield(trial,'spikes');

%%

for u = 1:1:98
    fr(u) = tuning_curve(trial,u,20,'density','no');
end

x = linspace(0,0.1,1000);
y = cell(98,8);
for u = 1:1:98
    for a = 1:1:8
        y{u,a} = makedist('Normal','mu',fr(u).mean_values(a),'sigma',fr(u).std_values(a));
    end
end
%%

test = trial(50,8).spikes;

rate = sum(test,2)/length(test);

for u = 1:1:98
    for a = 1:1:8
        prob(u,a) = pdf(y{u,a},rate(u));
    end
end

[~,idx] = max(prob,[],2);

angle = mean(idx);



