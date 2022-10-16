%% Experiment 1: Matrix Completion with RMSE Loss
clear
addpath Functions

loader = load('Data/MAT_30.mat');
MW = loader.MW; 
MI = loader.MI;
r = 3;
epochs = 200;
lossfun = 'RMSE';

%%%%%%%%%%%%%%%%%%%%%%%%%%% Well-Conditioned %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); learning_rate = 0.3; 
[~, fscsgd_well] = scaledsgd(MW, r, epochs, learning_rate, lossfun, true);

rng(1); learning_rate = 0.3; 
[~, fsgd_well]   = scaledsgd(MW, r, epochs, learning_rate, lossfun, false);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ill-Conditioned %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); learning_rate = 0.3; 
[~, fscsgd_ill]  = scaledsgd(MI, r, epochs, learning_rate, lossfun, true);

rng(1); learning_rate = 0.3; 
[~, fsgd_ill]    = scaledsgd(MI, r, epochs, learning_rate, lossfun, false);

%%%%%%%%%%%%%%%%%%%%%%%% Plot ScaledSGD vs SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotMAT(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill);