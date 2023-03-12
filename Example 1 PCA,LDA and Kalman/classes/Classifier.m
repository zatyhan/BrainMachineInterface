classdef Classifier
    %CLASSIFIER Class of classifier classes
    
    properties
        NN
        LDA1
        LDA2
        LDA3
        SVM
        NB
        ECOC
    end
    
    methods
        function obj = Classifier()
            %CLASSIFIER Construct an instance of this class
            
            obj.NN = nnClassifier();
            obj.LDA1 = ldaClassifier(1);
            obj.LDA2 = ldaClassifier(2);
            obj.LDA3 = ldaClassifier(3);
            obj.SVM = svmClassifier();
            obj.NB = nbClassifier();
            obj.ECOC = ecocClassifier();
        end
    end
end

