%% Ruben Ruiz-Mateos Serrano, Start date:12-02-2021

function [time,RMSE] = evaluate(fun,par,x,y)
    % fun - function handle, @algorithm_name
    % par - struct containing all required parameters, struct.par
    % x - training input data
    % y - training output data
    
    tStart = tic;   %Measure time
    
    [y_hat,~] = fun(x,par);     %Evaluate function
    
    time = toc(tStart);    %Return time
    
    RMSE = sqrt((norm(y(:)-y_hat(:),2).^2)/numel(y));    %Calculate RMSE    
end