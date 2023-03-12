function [] = raster_plot(data,sel,opt,type)
    % [] = RASTER_PLOT(data,sel,opt,type)
    % data - given struct array 
    % sel - struct with fields:
        %.trial - range or single trial value
        %.angle - range or single angle value
        %.unit - range or single unit value
    % opt - string argument: 
        % 'trial' - displays trials in array sel 
        % for same unit and angle
        % 'unit' - displays units in array sel
        % for same trial and angle
        % 'angle' - displays angles in array sel
        % for same unit and trial  
    % type - string argument: 
        % 'dot' - displays colour scatter plot
        % 'line' - displays monocromatic line plot
        
    hold on;
    if strcmpi(opt,'trial')
       if length(sel.unit)~=1 || length(sel.angle)~=1  
          error('Invalid selection: only one unit and angle');
       end
       if strcmpi(type,'dot')
            for i = 1:1:length(sel.trial)
                plot_data = data(sel.trial(i),sel.angle).spikes(sel.unit,:);
                line([plot_data(i),plot_data(i)],[0,1]);
                plot_data(plot_data==0) = NaN;
                scatter(1:1:length(plot_data),sel.trial(i)*plot_data,'filled');
            end
            ylim([sel.trial(1)-1,sel.trial(end)]);
            set(gca,'FontSize',15);
            xlabel('Time [ms]','FontSize',20);
            ylabel('Trials','FontSize',20);
        elseif strcmpi(type,'line')
            t = zeros(1,1000,length(sel.trial));
            for i = 1:1:length(sel.trial)
                t(:,:,i) = [data(sel.trial(i),sel.angle).spikes(sel.unit,:),zeros(1,1000-length(data(sel.trial(i),sel.angle).spikes(sel.unit,:)))];
            end
            plot_data = transpose(repmat(1:1:1000,length(sel.trial),1)).*squeeze(t);
            for i = 1:1:1000
                for ii = 1:1:length(sel.trial)
                line([plot_data(i,ii),plot_data(i,ii)],[sel.trial(ii)-1,sel.trial(ii)],'Color','b')
                end
            end
            ylim([sel.trial(1)-1,sel.trial(end)]);
            set(gca,'FontSize',15);
            xlabel('Time [ms]','FontSize',20);
            ylabel('Trials','FontSize',20);
       else 
           error('Invalid option: must be *dot* or *line*');
       end
    elseif strcmpi(opt,'unit')
        if length(sel.trial)~=1 || length(sel.angle)~=1  
           error('Invalid selection: only one trial and angle');
        end
        if strcmpi(type,'dot')
            for i = 1:1:length(sel.unit)
                plot_data = data(sel.trial,sel.angle).spikes(sel.unit(i),:);
                plot_data(plot_data==0) = NaN;
                scatter(1:1:length(plot_data),sel.unit(i)*plot_data,'filled');
            end
            ylim([sel.unit(1)-1,sel.unit(end)]);
            set(gca,'FontSize',15);
            xlabel('Time [ms]','FontSize',20);
            ylabel('Units','FontSize',20);
        elseif strcmpi(type,'line')
            t = zeros(1,1000,length(sel.unit));
            for i = 1:1:length(sel.unit)
                t(:,:,i) = [data(sel.trial,sel.angle).spikes(sel.unit(i),:),zeros(1,1000-length(data(sel.trial,sel.angle).spikes(sel.unit(i),:)))];
            end
            plot_data = transpose(repmat(1:1:1000,length(sel.unit),1)).*squeeze(t);
            for i = 1:1:1000
                for ii = 1:1:length(sel.unit)
                line([plot_data(i,ii),plot_data(i,ii)],[sel.unit(ii)-1,sel.unit(ii)],'Color','b')
                end
            end
            ylim([sel.unit(1)-1,sel.unit(end)]);
            set(gca,'FontSize',15);
            xlabel('Time [ms]','FontSize',20);
            ylabel('Units','FontSize',20);
        else 
           error('Invalid option: must be *dot* or *line*');
        end
    elseif strcmpi(opt,'angle')
        if length(sel.trial)~=1 || length(sel.unit)~=1  
           error('Invalid selection: only one trial and unit');
        end
        if strcmpi(type,'dot')
            for i = 1:1:length(sel.angle)
                plot_data = data(sel.trial,sel.angle(i)).spikes(sel.unit,:);
                plot_data(plot_data==0) = NaN;
                scatter(1:1:length(plot_data),sel.angle(i)*plot_data,'filled');
            end
            ylim([sel.angle(1)-1,sel.angle(end)]);
            set(gca,'FontSize',15);
            xlabel('Time [ms]','FontSize',20);
            ylabel('Units','FontSize',20);
        elseif strcmpi(type,'line')
            t = zeros(1,1000,length(sel.angle));
            for i = 1:1:length(sel.angle)
                t(:,:,i) = [data(sel.trial,sel.angle(i)).spikes(sel.unit,:),zeros(1,1000-length(data(sel.trial,sel.angle(i)).spikes(sel.unit,:)))];
            end
            plot_data = transpose(repmat(1:1:1000,length(sel.angle),1)).*squeeze(t);
            for i = 1:1:1000
                for ii = 1:1:length(sel.angle)
                line([plot_data(i,ii),plot_data(i,ii)],[sel.angle(ii)-1,sel.angle(ii)],'Color','b')
                end
            end
            ylim([sel.angle(1)-1,sel.angle(end)]);
            set(gca,'FontSize',15);
            xlabel('Time [ms]','FontSize',20);
            ylabel('Units','FontSize',20);
        else 
           error('Invalid option: must be *dot* or *line*');
        end 
    else
        error('Invalid option: must be *unit*, *trial* or *angle*');
    end
end


