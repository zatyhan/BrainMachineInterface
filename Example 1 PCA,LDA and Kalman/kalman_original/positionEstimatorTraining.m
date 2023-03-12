function [model] = positionEstimatorTraining(training_data)
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  A_npcr = 4;
  H_npcr = 4;
  model = KalmanModelOriginal();
  model = model.fit(training_data, A_npcr, H_npcr);
  
  % Return Value:
  
  % - model object:
  %     single model object that represents the trained model with
  %     appropriate model parameters
  
end