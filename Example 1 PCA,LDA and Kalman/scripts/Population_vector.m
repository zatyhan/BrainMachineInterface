%% POPULATION VECTOR

set(groot, 'defaultTextInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
clc; clear variables; close all;

addpath('funcs','data');

load('monkeydata_training.mat');

%Obtain preferred angles based on PSTH

for j = 1:1:98
    opt.unit = j;
    for i = 1:1:8
        opt.angle = i;
        h = PSTH(trial,opt,5,'no');
        ylim([0,23]);
        s(i) = sum(h.psth);
    end

    [~,out] = sort(s);
    unit(j).angle_order = flip(out);
end

close all;

%Obtain preferred angles based on tuning curve

for u = 1:1:98
    out = tuning_curve(trial,u,20,'density','no');
    Fit(u,:) = out.Fit;  
end

%% Data preprocessing

pref = extractfield(unit,'angle_order');
pref = pref(1,:);

data_spikes = extractfield(section(trial),'spikes');
data_pos = extractfield(section(trial),'handPos');
data = trial;

%% Population vector

close all; clc;

angle_list = [30/180*pi,70/180*pi,110/180*pi,150/180*pi,190/180*pi,230/180*pi,310/180*pi,350/180*pi];

figure;
b = 1;
for t = 1:1:100
    for a = 1:1:8
        clearvars -except pref data_spikes data_pos unit trial data t a angle_list Fit b RMSE
        
        var = data_spikes(1,:,t,a);
        l = length(var(~isnan(var)));
        spikes = data_spikes(:,1:l,t,a);
        pos = data_pos(1:2,1:l,t,a);

        init_pos = pos(:,1);
        position(:,1) = init_pos;

        % Determine baseline firing rate of all units from samples 1 to 300

        base_data = data(t,a).spikes;
        for u = 1:1:98
            base_counts(u,:) = histcounts(base_data(u,:).*[1:1:length(base_data)],[1:20:length(base_data)]);
            base_fr = base_counts/20;
        end

        b_fr = mean(base_fr,2);

        acc = 2;
        for n = 1:20:length(spikes)-20
            s = spikes(:,n:n+20);
            counts = sum(s,2);
            fr = counts/20;
            % If angle is to be calculated with tuning curve
            x = linspace(0,2*pi,100);
            for i = 1:1:98
                [~, min_idx] = min(abs(Fit(i,:)-(fr(i)+b_fr(i))));
                angles(i) = x(min_idx);
            end
%             [p(1,:),p(2,:)] = pol2cart(angle_list(pref)',fr./(b_fr+eps)); % avoid NaN
            [p(1,:),p(2,:)] = pol2cart(angles',b_fr./(fr+b_fr+eps)); % avoid NaN
%             figure;
            pred_motion = sum(p,2);
%             c = compass([p(1,:),pred_motion(1)],[p(2,:),pred_motion(2)],'b',2);
%             c1 = c(end);
%             c1.LineWidth = 2;
%             c1.Color = 'g';
            position(:,acc) = init_pos+pred_motion*0.3;
            init_pos = position(:,acc);
            acc = acc + 1;
        end

        plot(pos(1,:),pos(2,:),'Color','b');
        hold on;
        plot(position(1,:),position(2,:),'Color','r');
        
        RMSE(b) = sqrt(immse(norm(pos),norm(position)));
        b = b+1;
    end
end

RMSE = mean(RMSE);

