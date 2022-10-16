%% Experiment 4: Matrix Completion with Pointwise Cross-entropy Loss
clear
addpath Functions

loader = load('Data/MAT_30.mat');
MW = loader.MW; 
MI = loader.MI;
r = 3;
epochs = 200;
lossfun = '1bit';

%%%%%%%%%%%%%%%%%%%%%%%%%%% Well-Conditioned %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); learning_rate = 1; 
[~, fscsgd_well] = scaledsgd(MW, r, epochs, learning_rate, lossfun, true);

rng(1); learning_rate = 1; 
[~, fsgd_well]   = scaledsgd(MW, r, epochs, learning_rate, lossfun, false);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ill-Conditioned %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); learning_rate = 1; 
[~, fscsgd_ill]  = scaledsgd(MI, r, epochs, learning_rate, lossfun, true);

rng(1); learning_rate = 1; 
[~, fsgd_ill]    = scaledsgd(MI, r, epochs, learning_rate, lossfun, false);

%%%%%%%%%%%%%%%%%%%%%%%% Plot ScaledSGD vs SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotMAT(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill);