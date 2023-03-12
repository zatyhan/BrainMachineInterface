function sim_mat = xcorr_similarity(data)
% sim_mat = XCORR_SIMILARITY(data) Computes cross-correlation dissimilarity for every neuron unit pair
   % sim_mat = xcorr_similarity(data,fs)
    % data - given struct array 
    % fs - time-series sampling frequency
    
    sel.trial = [1:100]; sel.fs = 1000;
    sim_mat = zeros(98, 98);
    %Computing dissimilarity for every neural unit pair
    for unit = 1:98
        sel.unit = unit;
        max_corr = zeros(8, 98);
        stdevs = zeros(8, 98);
        for angle = 1:8
            sel.angle = angle;
            subplot(2, 4, angle);
            [map, mean_spikes] = xcorrmap(data, sel, false);
            max_corr(angle, :) = max(map, [], 2);
            stdevs(angle, :) = std(mean_spikes, [], 2);
        end
        % Determine largest magnitude angle and save correlation signatures
        [~,max_angle] = max(mean(max_corr, 2));
        % stdev of current unit
        sel.angle = max_angle;
        [~, mean_spikes] = xcorrmap(data, sel, false);
        std_unit = std(mean_spikes, [], 2);
        sim_mat(unit, :) = max_corr(max_angle, :)...
            ./(stdevs(max_angle, :).*std_unit'); 
        sim_mat(unit, max_corr(max_angle, :) == 0) = 0; % if a signal is zero, no correlation
    end
 
end

