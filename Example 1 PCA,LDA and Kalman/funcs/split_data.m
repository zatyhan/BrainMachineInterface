function [split_X,split_labels] = split_data(X,labels,fraction)
    % [split_X,split_labels] = SPLIT_DATA(X,labels,fraction)
    % X - samples*features matrix
    % labels - angle number column vector (includes all angles)
    % fraction - split percentage
    % split_X - struct fields:
        % .train
        % .test
    % split_labels - struct fields:
        % .train
        % .test
    
    l = length(X);
    r = randperm(l,l*fraction);
    split_X.train = X(r,:);
    split_labels.train = labels(r,:);
    
    split_X.test = X(setdiff(1:1:l,r),:);
    split_labels.test = labels(setdiff(1:1:l,r),:);
    
end

