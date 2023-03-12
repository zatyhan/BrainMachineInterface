function test = section(trial)
    % test = SECTION(trial)
    % data - given struct array 
    % test - useful section of data
        
    for i = 1:1:100
        for j = 1:1:8
            test(i,j).spikes = trial(i,j).spikes(:,300:end-100);
            test(i,j).handPos = trial(i,j).handPos(:,300:end-100);
        end
    end
end

