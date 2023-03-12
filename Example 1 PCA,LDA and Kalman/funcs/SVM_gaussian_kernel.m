function predict = SVM_gaussian_kernel(fr_avg,labels,C,s)
    % predict = SVM_GAUSSIAN_KERNEL(X,labels,C,sigma)
    % fr_avg - samples*features matrix
    % labels - angle number column vector (includes all angles)
    % C - regularization (SVM parameter)
    % s - variance (SVM parameter)
    
    len = size(fr_avg,1);
    T = len/8;
    
    % CLASSIFICATION 1
    % right(0): 3, 4, 5, 6 - left(1): 1, 2, 7, 8
    l = or((labels<=2),(labels>=7));
    model1 = svmTrain(fr_avg, double(l), C, @(x1, x2) gaussianKernel(x1, x2, s));
    
    % CLASSIFICATION 2
    % 3, 4 (1) - 5, 6 (0)
    l2_low = labels(~l);
    l2_low_log = labels(l2_low)<5;
    idx = ~l;
    model2_low = svmTrain(fr_avg(idx,:), double(l2_low_log), C, @(x1, x2) gaussianKernel(x1, x2, s));
    
    % 1, 2 (1) - 7, 8 (0)
    l2_high = labels(l);
    l2_high_log = labels(l2_high)<3;
    idx = l;
    model2_high = svmTrain(fr_avg(idx,:), double(l2_high_log), C, @(x1, x2) gaussianKernel(x1, x2, s));
    
    % CLASSIFICATION 3
    % 1 (1) - 2 (0)
    l3_high_low = l2_high(l2_high_log); 
    l3_high_low_log = l2_high(l3_high_low)<2;
    idx = labels<3;
    model3_high_low = svmTrain(fr_avg(idx,:), double(l3_high_low_log), C, @(x1, x2) gaussianKernel(x1, x2, s));
    LDA_1_2 = fitcknn(fr_avg(idx,:),double(l3_high_low_log));
    
    % 7 (1) - 8 (0)
    l3_high_high = l2_high(~l2_high_log); 
    l3_high_high_log = l2_high(l3_high_high)<8;
    idx = labels>6;
    model3_high_high = svmTrain(fr_avg(idx,:), double(l3_high_high_log), C, @(x1, x2) gaussianKernel(x1, x2, s));
    LDA_7_8 = fitcknn(fr_avg(idx,:),double(l3_high_high_log));
    
    % 3 (1) - 4 (0)
    l3_low_low = l2_low(l2_low_log);
    l3_low_low_log = l3_low_low<4;
    idx = and((labels>2),(labels<5));
    model3_low_low = svmTrain(fr_avg(idx,:), double(l3_low_low_log), C, @(x1, x2) gaussianKernel(x1, x2, s));
    LDA_3_4 = fitcknn(fr_avg(idx,:),double(l3_low_low_log));
    
    % 5 (1) - 6 (0)
    l3_low_high = l2_low(~l2_low_log); 
    l3_low_high_log = l3_low_high<6;
    idx = and((labels>4),(labels<7));
    model3_low_high = svmTrain(fr_avg(idx,:), double(l3_low_high_log), C, @(x1, x2) gaussianKernel(x1, x2, s));
    LDA_5_6 = fitcknn(fr_avg(idx,:),double(l3_low_high_log));
    
    predict.model1_3456_1278 = model1;
    predict.model2_34_56 = model2_low;
    predict.model2_12_78 = model2_high;
    predict.model3_1_2 = model3_high_low;
    predict.LDA_1_2 = LDA_1_2;
    predict.model3_7_8 = model3_high_high;
    predict.LDA_7_8 = LDA_7_8;
    predict.model3_3_4 = model3_low_low;
    predict.LDA_3_4 = LDA_3_4;
    predict.model3_5_6 = model3_low_high;
    predict.LDA_5_6 = LDA_5_6;
    
end