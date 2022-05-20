%% Load Data (Square Loss)
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
m = n^2;
spmat_SynWell30 = sampling(M_SynWell30,m);
spmat_SynIll30 = sampling(M_SynIll30,m);

epochs = 500;
learning_rate = 0.2; 
lossfun = 'square';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fsgdwell, ~] = psd_sgd(spmat_SynWell30, r, epochs, learning_rate, [], [], X0, lossfun);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fsgdill, ~] = psd_sgd(spmat_SynIll30, r, epochs, learning_rate, [], [], X0, lossfun);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fscsgdwell, ~] = psd_scalesgd(spmat_SynWell30, r, epochs, 1.5*learning_rate, [], [], X0, lossfun);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fscsgdill, ~] = psd_scalesgd(spmat_SynIll30, r, epochs, 1.5*learning_rate, [], [], X0, lossfun);fprintf('\n')

% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = 50;
plotfig(fscsgdwell,fsgdwell,fscsgdill,fsgdill,xlimit);

%% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = 50;
plotfig(fscsgdwell,fsgdwell,fscsgdill,fsgdill,xlimit);