% RANDOM SEED RMSE CALCULATOR

RMSE = zeros(1,50);
t = zeros(1,50);
percentage = zeros(1,50);
f = waitbar(0,'Processing...');
for itr = 1:50
    
    [RMSE(itr),t(itr),modelParameters] = testFunction_for_students_MTb('linear_regressor',true,true);
    percentage(itr) = modelParameters.percentage/modelParameters.count;
    waitbar(itr/50,f,'Processing...');
end
close(f);
beep;