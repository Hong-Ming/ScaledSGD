%% Load Data (Square Loss)
clc;clear
addpath Functions
% 30 x 30 symmetric well condtioned matrix, rank = 3
loader = load('Data/SYN_WELL30.mat'); 
MW_true = loader.M;

% 30 x 30 symmetric ill condtioned matrix, rank = 3
loader = load('Data/SYN_ILL30.mat'); 
MI_true = loader.M;

n = 30;
r = 3;

% Sample Ground Truth
percent = 100;
MW = sampling(MW_true,percent);
MI = sampling(MI_true,percent);

epochs = 500;
lr = 0.1; 
lossfun = 'square';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1);
% [~, fsgdwell] = scaledsgd(MW, r, epochs, lr, lossfun);fprintf('\n')

[~, fsgdwell] = temp(MW, r, epochs, lr, lossfun);fprintf('\n')
