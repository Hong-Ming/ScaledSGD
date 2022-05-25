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

%% Show SGD works well for well conditon, ScaleSGD workds well for both well and ill condition (n^2 samples)

% Sample Ground Truth
percent = 100;
MW = sampling(MW_true,percent);
MI = sampling(MI_true,percent);

epochs = 500;
lr = 0.3; 
lossfun = 'square';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fsgdwell, ~] = psd_sgd(MW, r, epochs, lr, lossfun, [], [], [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fsgdill, ~] = psd_sgd(MI, r, epochs, lr, lossfun, [], [], [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fscsgdwell, ~] = psd_scalesgd(MW, r, epochs, 1.5*lr, lossfun, [], [], [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fscsgdill, ~] = psd_scalesgd(MI, r, epochs, 1.5*lr, lossfun, [], [], [], X0);fprintf('\n')

% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = 50;
plotfig(fscsgdwell,fsgdwell,fscsgdill,fsgdill,xlimit);

%% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = 50;
plotfig(fscsgdwell,fsgdwell,fscsgdill,fsgdill,xlimit);