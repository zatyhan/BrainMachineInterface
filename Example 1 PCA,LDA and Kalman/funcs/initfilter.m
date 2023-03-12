function data_out = initfilter(data,J,overlap,window)
    % data_out = INITFILTER(data,J,overlap,window)
    % data - given struct array 
    % J - number of Hanning windows (new number of columns)
    % overlap - percentage of overlap between windows (decimals)
    % data_out - filtered output in the same format as data 
    % window - type of window specified as a string
    
    [T,A] = size(data); % get data size

    data_cell = struct2cell(data); % convert to cell matrix
    max_length = @(x) length(x); % function handle to get max length of all elements in cell
    l = cellfun(max_length,data_cell);
    N = max(squeeze(l(3,:,:)),[],'all'); % get maximum length
   
    L = ceil(N/(0.5+(1-overlap)*(J-1))); % determine window size
    
    H = zeros(J,N);
    for j = 1:J % create matrix with shifted versions of given window defined by overlap
        shift = round((j-1)*(1-overlap)*L);
        H(j,1:L+shift) = [zeros(1,shift) eval(sprintf('%s(L)',window))'];
    end
    
    H(:,1:round(L/2)) = []; % neglect first L/2 values
    
    for a = 1:A % apply kernel to spikes data for smoothing 
        for t = 1:T
            for u = 1:98
                var = data(t,a).spikes(u,:)';
                new_var = H*[var;zeros(length(H)-length(var),1)];
                data_out(t,a).spikes(u,:) = new_var';
            end
        end
    end

end