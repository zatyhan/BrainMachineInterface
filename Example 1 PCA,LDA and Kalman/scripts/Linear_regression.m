%% LINEAR REGRESSION
clearvars -except kin rate; clc;

load('monkeydata_training');

% Linear regressor

angle = 3;

% Train data

for t=1:70
    pos(t,:,:) = trial(t,angle).handPos(1:2,301:20:560)-trial(t,angle).handPos(1:2,301);
end
mean_pos = squeeze(mean(pos,1));
mean_pos = mean_pos';

for t = 1:1:70
    var = trial(t,angle).spikes(:,301:560);
    var(var==0)=NaN;
    edges = 0:20:length(var);
    for u = 1:1:98
        s(t,u,:) = histcounts(var(u,:).*[1:1:length(var(u,:))],edges);
    end
end
sum_s = squeeze(mean(s,1))';

% for u = 1:98
%     scatter(mean_pos(2,:),sum_s(:,u)');
%     f = polyfit(mean_pos(2,:),sum_s(:,u)',1);
%     f = polyval(f,mean_pos(2,:));
%     hold on;
%     plot(mean_pos(2,:),f);
%     pause(0.1);
%     hold off;
% end

s_unit = sum_s(:,1);
y = mean_pos(:,2);
p = polyfit(s_unit,y,1);
s_unit = [s_unit ones(length(s_unit),1)];
f = inv(s_unit'*s_unit)*s_unit'*y;

units = 1:10;
% units = [1:37 39:48 50:98];
% units = [1:30 32 34:37 40:43 45:48 50:51 53:68 70:72 75 77:83 85:92 94:98];

numBin = length(mean_pos);
yTrain = mean_pos(1:numBin,1:2);
sTrain = sum_s(1:numBin,units);
sTrain = [sTrain ones(numBin,1)];
f = inv(sTrain'*sTrain)*sTrain'*yTrain;

% Test data
clear pos s;

for t=1:30
    real_pos(t,:,:) = trial(t+70,angle).handPos(1:2,301:560)-trial(t+70,angle).handPos(1:2,301);
    pos(t,:,:) = trial(t+70,angle).handPos(1:2,301:20:560);
end
mean_pos_test = squeeze(mean(pos,1));
mean_pos_test = mean_pos_test';

for t = 1:30
    var = trial(t+70,2).spikes(:,301:560);
    var(var==0)=NaN;
    edges = 0:20:length(var);
    for u = 1:1:98
        s(t,u,:) = histcounts(var(u,:).*[1:1:length(var(u,:))],edges);
    end
end
sum_s_test = squeeze(mean(s,1))';

numBin = length(mean_pos_test);
sTest = [sum_s_test(1:numBin,units) ones(numBin,1)];
yActual = mean_pos_test(1:numBin,1:2);
yFit = sTest*f;

figure;
subplot(1,2,1);
for t = 1:30
    plot(301:560,squeeze(real_pos(t,:,:)));
    hold on;
end
plot(301:20:560,yFit);

subplot(1,2,2);
for t = 1:30
    plot(squeeze(real_pos(t,1,:)),squeeze(real_pos(t,2,:)));
    hold on;
end
plot(yFit(:,1),yFit(:,2),'Color','r','LineWidth',2);
