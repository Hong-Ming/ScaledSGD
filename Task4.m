%% Load Data (Pointwise Cross Entropy Loss Noisy Setting)
clc;clear
addpath Functions
% 30 x 30 symmetric well condtioned matrix, rank = 3
loader = load('Data/SYN_WELL_Noise30.mat'); 
MW_true = loader.M;

% 30 x 30 symmetric ill condtioned matrix, rank = 3
loader = load('Data/SYN_ILL_Noise30.mat'); 
MI_true = loader.M;

% Sample Ground Truth
percent = 100;
MW = sampling(MW_true,percent);
MI = sampling(MI_true,percent);
r = 5;
epochs = 300;
learning_rate = 0.01; 
lossfun = '1bit';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fsgdwell] = sgd(MW, r, epochs, learning_rate, lossfun);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fsgdill] = sgd(MI, r, epochs, learning_rate, lossfun);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fscsgdwell] = scaledsgd(MW, r, epochs, 15*learning_rate, lossfun);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fscsgdill] = scaledsgd(MI, r, epochs, 15*learning_rate, lossfun);

% compute noise floor
[U,S] = eig(MW_true); [~,idx] = sort(diag(S),'descend'); perm = idx(1:r);
U = U(:,perm); S = S(perm,perm); 
nfw = (1/2)*(1/numel(MW_true))*norm(MW_true-U*S*U','fro')^2;

[U,S] = eig(MI_true); [~,idx] = sort(diag(S),'descend'); perm = idx(1:r);
U = U(:,perm); S = S(perm,perm); 
nfi = (1/2)*(1/numel(MI_true))*norm(MI_true-U*S*U','fro')^2;

% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = inf;
plotfig1(fscsgdwell,fsgdwell,fscsgdill,fsgdill,nfw,nfi,xlimit);

