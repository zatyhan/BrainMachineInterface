%% Determining linearities and collinearities, 
% as well as possible transformations that might make the data a better fit
clc; clear variables; close all;

addpath('funcs','data');

load monkeydata0;

% Visualizing and understanding correlations between firing rate in
% previous 20 samples and displacement (all angles)

[X, Y] = extract_supervised(trial);
for i = 1:10
    figure(i); fig_counter = 1;
    for j = 1:10
        fig_counter
        subplot(10, 2, fig_counter);
        scatter(X((i-1)*10 + j, :), Y(1, :))
        title(sprintf('Unit %d', fig_counter)); xlabel('Average Firing Rate'); ylabel('X-displacement');
        subplot(10, 2, fig_counter + 1);
        scatter(X((i-1)*10 + j, :), Y(2, :))
        title(sprintf('Unit %d', fig_counter + 1)); xlabel('Average Firing Rate'); ylabel('Y-displacement');
        fig_counter = fig_counter +2;
        
        if (i-1)*10 + j == 98 % Completed all units
            break
        end
    end
end