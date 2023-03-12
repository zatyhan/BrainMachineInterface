# Brain Machine Interfaces
This repository contains our work for the Imperial College London "BMIS" module (2020-2021).

## Task
We were provided with a dataset of multi-channel ECoG recordings from a monkey during a reaching task. With this dataset, we carried out a neural decoding task, aiming to predict the dynamics of the monkey's arm, as well as to classify the direction of reach of the monkey (from 8 possible targets) in real-time.

## Approach
A requirement of the task was for the algorithms to be written in MATLAB. Our approach involved computing the average spike rate during movement preparationa and movement initiation separately. From these spike rates, PCA was used to reduce collinearity and the dimension of the data, followed by an LDA classifier for the target location. For decoding the dynamics of the monkey's arm, a Kalman filter algorithm was implemented. 
