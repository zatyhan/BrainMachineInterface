function [X, Y] = extract_supervised(data)
    % [X, Y] = EXTRACT_SUPERVISED(data) extracts data in the appropriate
    % format for supervised learning
    % Inputs:
    % data - given struct array 
    % Outputs:
    % Y: x and y positions for each given example
    % X: Average sampling rate for each given example
 
    X = []; Y = [];
    t_counter = 1;
    
    for T = 1:50
        for A = 1:8
            fprintf('Trial %d \n', t_counter);
            t_counter = t_counter + 1;
            for t = 320:20:length(data(T, A).spikes)
                if t == 320
                    Y = [Y, data(T, A).handPos(1:2, 320)...
                        - data(T, A).handPos(1:2, 1)];
                else
                     Y = [Y, data(T, A).handPos(1:2, t)...
                         - data(T, A).handPos(1:2, t-19)];
                end
                X = [X, mean(data(T, A).spikes(:, 300:t), 2)];
            end
        end
    end
end


