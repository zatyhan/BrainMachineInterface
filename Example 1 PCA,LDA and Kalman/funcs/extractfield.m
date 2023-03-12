function data_out = extractfield(data,field)
    % [] = EXTRACTFIELD(data,field)
        % data - given struct array 
        % field - string argument name of field
        % data_out - extracted field as a matrix or tensor (check new
        % dimension order)
    
    [R,C] = size(data);
    fields = fieldnames(data);
    
    if ~any(strcmp(fields,field))
        error('Invalid syntax: *field* must represent a field present in the struct');
    else
        for i = 1:1:R
            for j = 1:1:C
                data_out(:,:,i,j) = [data(i,j).(field),NaN(size(data(i,j).(field),1),1000-length(data(i,j).(field)))];
            end
        end
        data_out = squeeze(data_out);
    end
end