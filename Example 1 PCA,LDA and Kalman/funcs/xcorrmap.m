function [map, mean_spikes] = xcorrmap(data,sel,show)
    % [map, mean_spikes] = XCORRMAP(data,sel,show)
    % data - given struct array 
    % sel - struct with fields:
        %.trial - range or single trial value
        %.angle - single angle value
        %.unit - single unit value
        %.fs - sampling frequency
    % show - boolean argument: 
        % true - computes cross-correlation heatmap between neuronal unit
        % and other units
        % false - does not compute/disply heatmap
        
     % Average over trials selected
     if length(sel.trial) ~= 1
         mean_spikes = data(sel.trial(1), sel.angle).spikes; % initial data at first trial
         for i = 2:length(sel.trial)
             new_spikes = data(sel.trial(i), sel.angle).spikes;
             mean_shape = size(mean_spikes); new_data_shape = size(new_spikes);
             % If some trials are longer, add zero padding to compute mean
             if mean_shape(2) > new_data_shape(2)
                 new_spikes = [new_spikes zeros(98, mean_shape(2) - new_data_shape(2))];
             elseif mean_shape(2) < new_data_shape(2)
                 mean_spikes = [mean_spikes zeros(98, new_data_shape(2) - mean_shape(2))];
             end
             mean_spikes = mean_spikes + new_spikes; % sum over all of them first
         end
         mean_spikes = mean_spikes/length(sel.trial);
     else
         mean_spikes = data(sel.trial, sel.angle).spikes;
     end
     spike_shape = size(mean_spikes);
     
     % Compute the cross-correlation map between neuronal units
     map = [];
     for i = 1:98
         map = [map; xcorr(mean_spikes(sel.unit, :), mean_spikes(i, :))];
     end
     
     % If selected, display correlation heatmap
     if show
         imagesc(map, [min(map,[],'all'), max(map,[],'all')]);
         xticklabels({(xticks - length(mean_spikes))/sel.fs})
         xlabel('Time lags (s)'); ylabel('Neural Unit Number');
         title(sprintf('Unit %d with other Units (angle %d, trials %d-%d)', ...
             sel.unit, sel.angle, sel.trial(1), sel.trial(end) ));
         colormap('hot');
         colorbar;
     end
end