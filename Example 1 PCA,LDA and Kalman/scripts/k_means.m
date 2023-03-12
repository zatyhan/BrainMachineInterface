%% k-means algorithm
clear variables; clc; close all;

addpath('funcs','data');

load('monkeydata_training');

[~,fr_avg] = fr_features(trial,80,560);
[~,score,~] = pca(fr_avg,'NumComponents',3);
[idx,mu] = kmeans(score,8);

c = repelem([1,2,3,4,5,6,7,8],100)';%idx;
scatter3(score(:,1),score(:,2),score(:,3),[],c,'filled');
hold on;
scatter3(mu(:,1),mu(:,2),mu(:,3),200,'x');
set(gca,'FontSize',15);
xlabel('Principal Component 1','FontSize',20);
ylabel('Principal Component 2','FontSize',20);
zlabel('Principal Component 3','FontSize',20);
text(score(:,1),score(:,2),score(:,3),cellstr(num2str(repelem([1,2,3,4,5,6,7,8],100)')));
col = colorbar;
col.Label.String = 'Angles';
col.Label.FontSize = 15;
col.Label.Interpreter = 'latex';

