function out = PSTH(data,sel,period,show)
    % out = PSTH(data,sel,period)
    % data - given struct array
    % sel - struct with fields:
        %.angle - single angle value
        %.unit - single unit value
    % period - time interval, must be multiple
    % of the number of samples per trial
    % show - string argument:
        %'show' - displays PSTH
    %out - struct with fields:
        %.psth = spike counts per bin
        %.bins = bin array
        
    [T,~] = size(data);
    N_array = zeros(1,T);
    for i = 1:1:T
        plot_data = data(i,sel.angle).spikes(sel.unit,:);
        N_array(i) = length(plot_data);
    end
    N = max(N_array);
    edges = 0:period:N;
    psth = zeros(length(edges)-1,1);
    
    for jj = 1:1:T
        var = data(jj,sel.angle).spikes(sel.unit,:);
        n = length(var);
        var(data(jj,sel.angle).spikes(sel.unit,:)==0) = NaN;
        psth = psth + histcounts([1:1:n].*var,edges)';
    end
    if strcmpi(show,'show')
        bar(period:period:length(psth)*period,psth','b');
    end
    set(gca,'FontSize',15);
    xlabel('Time [ms]','FontSize',20);
    ylabel('\# of spikes','FontSize',20);  
    
    out.psth = psth';
    out.bins = period:period:length(psth)*period;
end
