%% Experiment 5: Matrix Completion with RMSE loss on Noisy Datasets
clear; addpath Functions
loader = load('Data/MAT_Noise_30.mat');
MW = loader.MW; 
MI = loader.MI;
r = 5;
epochs = 50;
lossfun = 'RMSE';

%%%%%%%%%%%%%%%%%%%%%%%%%%% Well-Conditioned %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); learning_rate = 0.15; 
[~, fscsgd_well] = scaledsgd(MW, r, epochs, learning_rate, lossfun, true);

rng(1); learning_rate = 0.01; 
[~, fsgd_well]   = scaledsgd(MW, r, epochs, learning_rate, lossfun, false);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ill-Conditioned %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); learning_rate = 0.15; 
[~, fscsgd_ill]  = scaledsgd(MI, r, epochs, learning_rate, lossfun, true);

rng(1); learning_rate = 0.01; 
[~, fsgd_ill]    = scaledsgd(MI, r, epochs, learning_rate, lossfun, false);

%%%%%%%%%%%%%%%%%%%%%%%% Plot ScaledSGD vs SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotMATNoise(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill,MW,MI,r)
save('Data/Results/EXP5.mat','fscsgd_well','fsgd_well','fscsgd_ill','fsgd_ill','MW','MI','r')