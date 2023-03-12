function F_data = F_analysis(trial,opt)
    % [] = F_ANALYSIS(trial)
        % trial - given struct array 
        % opt - string argument:
            % detrend - fft of detrended data
            % [] - fft of raw data
        % F_data - Fourier transform of data
    
    data = section(trial);   
    data = extractfield(data,'spikes');

    [U,~,T,A] = size(data);

    F_data = NaN(T,A,U,1000);
    for i = 1:1:T
        for j = 1:1:A
            var = data(1,:,i,j);
            nonan_len = length(var(~isnan(var)));
            for k = 1:1:U
                if strcmp(opt,'detrend')
                    x = detrend(squeeze(data(k,1:1:nonan_len,i,j)));
                end
                F_data(i,j,k,1:nonan_len) = abs(fftshift(fft(squeeze(x))));
            end
        end
    end
end

