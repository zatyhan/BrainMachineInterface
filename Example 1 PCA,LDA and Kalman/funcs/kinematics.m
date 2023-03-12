function [x,y,x_avg,y_avg,x_vel,y_vel,x_acc,y_acc,l] = kinematics(data)
    % KINEMATICS Calculates multiple kinematic variables
    % data - given struct array
    % x - x position extended to maximum length (assuming stationarity) 
    % y - y position extended to maximum length (assuming stationarity) 
    % x_avg - mean x position extended to maximum length (assuming stationarity) 
    % y_avg - mean y position extended to maximum length (assuming stationarity) 
    % x_vel - x velocity extended to maximum length (assuming stationarity) 
    % y_vel - y velocity extended to maximum length (assuming stationarity)
    % x_acc - x acceleration extended to maximum length (assuming stationarity)
    % y_acc - y acceleration extended to maximum length (assuming stationarity)  
    % l - time length (N) of each trial
    
    data_cell = struct2cell(data); % convert to cell matrix
    max_length = @(x) length(x); % function handle to get max length of all elements in cell
    l = cellfun(max_length,data_cell);
    l = squeeze(l(3,:,:)); % retain only handPos information
    
    L = max(l,[],'all');
    
    [T,A] = size(data); % get dimensions of data
    x_avg = zeros(A,L); % initialise variables
    y_avg = zeros(A,L);
    x = zeros(T,A,L);
    y = zeros(T,A,L);
    x_vel = zeros(T,A,L);
    y_vel = zeros(T,A,L);
    x_acc = zeros(T,A,L);
    y_acc = zeros(T,A,L);
    for a = 1:1:A
        for t = 1:1:T
            var_x = data(t,a).handPos(1,:);
            var_y = data(t,a).handPos(2,:);
            x(t,a,:) = [var_x var_x(end)*ones(1,L-length(var_x))];
            y(t,a,:) = [var_y var_y(end)*ones(1,L-length(var_y))];
            x_vel(t,a,:) = [0 diff(squeeze(x(t,a,:))')/0.02]; %calculate immediate velocity
            y_vel(t,a,:) = [0 diff(squeeze(y(t,a,:))')/0.02];
            x_acc(t,a,:) = [0 0 diff(diff(squeeze(x(t,a,:))')/0.02)/0.02]; %calculate immediate acceleration
            y_acc(t,a,:) = [0 0 diff(diff(squeeze(y(t,a,:))')/0.02)/0.02];
        end
        x_avg(a,:) = squeeze(mean(x(:,a,:),1))';
        y_avg(a,:) = squeeze(mean(y(:,a,:),1))';
    end
    
end

