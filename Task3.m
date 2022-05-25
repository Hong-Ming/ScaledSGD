%% Load Data (Pairwise Square Loss)
clc;clear
addpath Functions
% 30 x 30 symmetric well condtioned matrix, rank = 3
loader = load('Data/SYN_WELL30.mat'); 
M_SynWell30 = loader.M;

% 30 x 30 symmetric ill condtioned matrix, rank = 3
loader = load('Data/SYN_ILL30.mat'); 
M_SynIll30 = loader.M;

n = 30;
r = 3;

%% Show SGD works well for well conditon, ScaleSGD workds well for both well and ill condition (n^2 samples)

% Sample Ground Truth
percent = 100;
spmat_SynWell30 = sampling(M_SynWell30,percent);
spmat_SynIll30 = sampling(M_SynIll30,percent);

rng(10)
epochs = 50;
lr = 0.05;  
lossfun = 'pair';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(10); X0 = randn(n,r);  
[~, fsgdwell, ~] = psd_sgd(spmat_SynWell30, r, epochs, lr, lossfun, [], [], [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(10); X0 = randn(n,r);  
[~, fsgdill, ~] = psd_sgd(spmat_SynIll30, r, epochs, lr, lossfun, [], [], [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(10); X0 = randn(n,r);  
[~, fscsgdwell, ~] = psd_scalesgd(spmat_SynWell30, r, epochs, 1*lr, lossfun, [], [], [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(10); X0 = randn(n,r);  
[~, fscsgdill, ~] = psd_scalesgd(spmat_SynIll30, r, epochs, 1*lr, lossfun, [], [], [], X0);fprintf('\n')

% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = 20;
plotfig(fscsgdwell,fsgdwell,fscsgdill,fsgdill,xlimit);

%% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = 20;
plotfig(fscsgdwell,fsgdwell,fscsgdill,fsgdill,xlimit);