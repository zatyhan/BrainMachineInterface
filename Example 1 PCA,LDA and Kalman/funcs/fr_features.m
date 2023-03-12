function [fr_total, fr_avg, X] = fr_features(data,dt,N)
    %FR_FEATURES Calculates the firing rate of the data in bins of size dt.
    % data - given data struct
    % dt - time bin size
    % N - total number of samples length of
    % fr_total - spiking rate divided in bins
    % fr_avg - average spiking rate across bins
    % X - average spiking rate across bins in different trial sections (prior to movement,
    % peri-movement and total)
    
    [T,A] = size(data); %get trial and angle length
    
    acc = 1;
    fr_avg = zeros(T*A,98); % initialise variables
    fr_avg1 = zeros(T*A,98); % initialise variables
    fr_avg2 = zeros(T*A,98); % initialise variables
    fr_total = zeros(T*A,N/dt*98);
    for t=1:1:T
        for a=1:1:A
            fr = zeros(98,length(0:dt:N)-1);
            fr1 = zeros(98,length(0:dt:N)-1);
            fr2 = zeros(98,length(0:dt:N)-1);
            for u=1:1:98
                var = data(t,a).spikes(u,1:N);
                var_alt = var;
                var_alt(var_alt==0) = NaN; % make zeros equal to NaN
                count = histcounts([1:1:N].*var_alt,0:dt:N); % count spikes in every dt bin until N
                fr(u,:) = count/dt;
                count1 = sum(var(1:200)); % count spikes in every dt bin until 200
                fr1(u,:) = count1/200;
                count2 = sum(var(250:320)); % count spikes in every dt bin from  250 to 320
                fr2(u,:) = count2/120;
            end
            fr_avg(acc,:) = mean(fr,2); % get mean firing rate across bins
            fr_avg1(acc,:) = mean(fr1,2); % get mean firing rate across bins
            fr_avg2(acc,:) = mean(fr2,2); % get mean firing rate across bins
            f = reshape(fr,size(fr,1)*size(fr,2),1);
            fr_total(acc,:) = f; % get all firing rates ordered in 98 blocks of the same bin
            acc = acc+1;
        end
    end
    X = [fr_avg,fr_avg1,fr_avg2];
end