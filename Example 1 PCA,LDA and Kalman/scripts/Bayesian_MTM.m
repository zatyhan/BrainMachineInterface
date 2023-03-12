%% MMT Bayesian filter
%% Ruben Ruiz-Mateos Serrano, Start date: 27/02/2021

set(groot, 'defaultTextInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
clear variables; clc; close all;

load('monkeydata_training');

Data = extractfield(trial,'handPos');

Spikes = extractfield(trial,'spikes');

%%

split_spikes = squeeze(Spikes(1,1:672,:,1));
data = Data(1:2,1:672,1,1);

vel(1,:) = [0,diff(data(1,:),1)];
vel(2,:) = [0,diff(data(2,:),1)];

acc(1,:) = [0,0,diff(data(1,:),2)];
acc(2,:) = [0,0,diff(data(2,:),2)];

for p = 1:1:length(vel)
    mag_pos(p) = norm(data(:,p));
    mag_vel(p) = norm(vel(:,p));
end

figure;
subplot(3,1,1);
plot(1:1:length(data),data,'LineWidth',2)
title('Position','FontSize',20)
subplot(3,1,2);
plot(1:1:length(data),vel,'LineWidth',2)
title('Velocity','FontSize',20)
subplot(3,1,3);
plot(1:1:length(data),acc,'LineWidth',2)
title('Acceleration','FontSize',20)

x = cat(1,data,vel,acc,mag_pos,mag_vel);

% Initialising variables
spikes = split_spikes(200:1:end-100,:);
X = x(:,100:1:end);
m = randi(8);
delta = 50;
clear frFit;

figure;
sel.angle = 1;
sel.unit = 1;
h = PSTH(trial,sel,delta,'show');
hold on

% Neural encoding

for lag = 0:1:200/delta
    for t = 1:1:100
        s = histcounts(spikes(:,t).*[1:1:length(spikes(:,t))]',1:delta:length(spikes(:,t)));
        numBins = length(s);
        kin = X(:,1:delta:length(X)-delta);
        predictor = [ones(numBins,1),kin(1,lag+1:lag+numBins)',kin(2,lag+1:lag+numBins)',kin(3,lag+1:lag+numBins)',kin(4,lag+1:lag+numBins)',kin(5,lag+1:lag+numBins)',kin(6,lag+1:lag+numBins)',kin(7,lag+1:lag+numBins)',kin(8,lag+1:lag+numBins)'];
        theta{t} = glmfit(predictor(:,2:end),s,'poisson');
        frFit(:,t) = exp(predictor*theta{t});
    end

    fr_pred = sum(frFit,2);

    bar(h.bins(200/delta+1:200/delta+length(fr_pred)),fr_pred);
end

for i = 1:1:8
    Pi{i} = rand(8,1);
    V{i} = rand(8);
    b{i} = rand(8,1);
    Q{i} = rand(8);
    A{i} = rand(8); 
end

for p = 1:1:8
    x1_m{1}(p) = makedist('Normal','mu',Pi{m}(p),'sigma',V{m}(p,p));
end

for t = 1:1:length(data)
    for p = 1:1:8
        x1_m{t+1}(p) = makedist('Normal','mu',A{m}(p,p)*random(x1_m{t}(p),1)+b{m}(p),'sigma',Q{m}(p,p));
    end
    
end




