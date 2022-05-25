%% Load Data (Asymmetric Square Loss)
clc;clear
addpath Functions
loader = load('Data/MOVIELENS.mat');
M_true = loader.M;

%% Show SGD works well for well conditon, ScaleSGD workds well for both well and ill condition (n^2 samples)

% Sample Ground Truth
r = 10;
n = size(M_true,1);
n1 = size(M_true,2);
percent = 50;
M = sampling(M_true,percent);

epochs = 100;
lossfun = 'square';
mo = 0;
mb = 500;

lr1 = 1;
lr2 = 2000;
reg = 1e-4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); U0 = rand(n,r)/sqrt(r); V0 = rand(n1,r)/sqrt(r);  
[~, ~, fsgd, ~, asgd] = asym_sgd(M, r, epochs, lr1, lossfun, mo, mb, reg, U0, V0, M_true);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); U0 = rand(n,r)/sqrt(r); V0 = rand(n1,r)/sqrt(r);  
[~, ~, fscsgd, ~, ascsgd] = asym_scalesgd(M, r, epochs, lr2, lossfun, mo, mb, reg, U0, V0, M_true);fprintf('\n')

% Plot ScaleSGD vs SGD
xlimit = inf;
plotfig1(fscsgd,fsgd,ascsgd,asgd,xlimit,percent);

%% Plot ScaleSGD vs SGD (Well condition and Ill condition)
xlimit = inf;
plotfig1(fscsgd,fsgd,ascsgd,asgd,xlimit,percent);

%% Test code

test = rand(3,1)*rand(1,5);
epochs = 5000;
lossfun = 'square';
lr1 = 0.1;
lr2 = 1;

% rng(1);
% [~, ~, fsgd, ~, asgd] = asym_sgd(test, 1, epochs, lr1, lossfun);
rng(1);
[~, ~, fsgd, ~, asgd] = asym_scalesgd(test, 1, epochs, lr2, lossfun);
