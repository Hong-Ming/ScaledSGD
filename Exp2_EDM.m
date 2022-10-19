%% Experiment 2: Euclidean Distance Matrix (EDM) Completion
clear; addpath('Functions')
loader = load('Data/EDM_30.mat'); 
DW = loader.DW; XW = loader.XW;
DI = loader.DI; XI = loader.XI;
r = 3;
epochs = 500;
lossfun = 'EDM';

%%%%%%%%%%%%%%%%%%%%%%%%%%% Well-Conditioned %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); learning_rate = 0.2; 
[~, fscsgd_well] = scaledsgd(DW, r, epochs, learning_rate, lossfun, true);

rng(1); learning_rate = 0.02; 
[~, fsgd_well]   = scaledsgd(DW, r, epochs, learning_rate, lossfun, false);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ill-Conditioned %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); learning_rate = 0.2; 
[~, fscsgd_ill]  = scaledsgd(DI, r, epochs, learning_rate, lossfun, true);

rng(1); learning_rate = 0.002; 
[~, fsgd_ill]    = scaledsgd(DI, r, epochs, learning_rate, lossfun, false);

%%%%%%%%%%%%%%%%%%%%%%%% Plot ScaledSGD vs SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotEDM(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill, XW, XI)
save('Data/Results/EXP2.mat','fscsgd_well','fsgd_well','fscsgd_ill','fsgd_ill', 'XW', 'XI')