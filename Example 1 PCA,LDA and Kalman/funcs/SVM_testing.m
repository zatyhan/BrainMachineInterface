function pred_angle = SVM_testing(fr_avg,model)
    % pred_angle = SVM_testing(model)
    % fr_avg - samples*features matrix
    % model - struct with trained SVM models
    % pred_angle - integer predicted angle
    
    pred_1 = svmPredict(model.model1_3456_1278,fr_avg(1,:));
    if pred_1 == 1 % left(1): 1, 2, 7, 8
        pred_2 = svmPredict(model.model2_12_78, fr_avg(1,:)); % 1, 2 (1) - 7, 8 (0)
        if pred_2 == 1 % 1 (1) - 2 (0)
            %pred_3 = svmPredict(model.model3_1_2, fr_avg(1,:));
            pred_3 = predict(model.LDA_1_2, fr_avg(1,:));
            if pred_3 == 1
                pred_angle = 1;
            else 
                pred_angle = 2;
            end
        elseif pred_2 == 0 % 7 (1) - 8 (0)
            %pred_3 = svmPredict(model.model3_7_8, fr_avg(1,:));
            pred_3 = predict(model.LDA_7_8, fr_avg(1,:));
            if pred_3 == 1 
                pred_angle = 7;
            else
                pred_angle = 8;
            end
        end
    elseif pred_1 == 0 % right(0): 3, 4, 5, 6 
        pred_2 = svmPredict(model.model2_34_56, fr_avg(1,:)); % 3, 4 (1) - 5, 6 (0)
        if pred_2 == 1 % 3 (1) - 4 (0)
            %pred_3 = svmPredict(model.model3_3_4, fr_avg(1,:));
            pred_3 = predict(model.LDA_3_4, fr_avg(1,:));
            if pred_3 == 1
                pred_angle = 3;
            else 
                pred_angle = 4;
            end
        elseif pred_2 == 0 % 5 (1) - 6 (0)
            %pred_3 = svmPredict(model.model3_5_6, fr_avg(1,:)); 
            pred_3 = predict(model.LDA_5_6, fr_avg(1,:)); 
            if pred_3 == 1 
                pred_angle = 5;
            else
                pred_angle = 6;
            end
        end
        
    end
    
end

