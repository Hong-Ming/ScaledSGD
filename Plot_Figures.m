%% Experiment 1: Matrix Completion with RMSE Loss
clear; addpath Functions; load('Data/Results/EXP1.mat')
plotMAT(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill)

%% Experiment 2: Euclidean Distance Matrix (EDM) Completion
clear; addpath Functions; load('Data/Results/EXP2.mat')
plotEDM(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill, XW, XI)

%% Experiment 3: Huge-scale Item-item Collaborative Filtering
clear; addpath Functions; load('Data/Results/EXP3.mat')
plotCFHuge(fscsgd,fsgd,aucscsgd,aucsgd,np_maximum)

%% Experiment 4: Matrix Completion with Pointwise Cross-entropy Loss
clear; addpath Functions; load('Data/Results/EXP4.mat')
plotMAT(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill)

%% Experiment 5: Matrix Completion with RMSE loss on Noisy Datasets
clear; addpath Functions; load('Data/Results/EXP5.mat')
plotMATNoise(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill,MW,MI,r)

%% Experiment 6: Matrix Completion with Pointwise Cross-entropy Loss on Noisy Datasets
clear; addpath Functions; load('Data/Results/EXP6.mat')
plotMATNoise(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill,MW,MI,r)

%% Experiment 7: Small-scale Item-item Collaborative Filtering
clear; addpath Functions; load('Data/Results/EXP7.mat')
plotCF(fscsgd,fsgd,aucscsgd,aucsgd,np_maximum)

%% Experiment 8: Medium-scale Item-item Collaborative Filtering
clear; addpath Functions; load('Data/Results/EXP8.mat')
plotCF(fscsgd,fsgd,aucscsgd,aucsgd,np_maximum)

%% Experiment 9: Large-scale Item-item Collaborative Filtering
clear; addpath Functions; load('Data/Results/EXP9.mat')
plotCF(fscsgd,fsgd,aucscsgd,aucsgd,np_maximum)
