%% Load Data (BPR Loss)
clc;clear
addpath Functions
loader = load('Data/MOVIELENS.mat');
M_true = loader.ItemItem;
% subsample item-item matrix
numnz = sum(M_true~=0,2);
[~,idx] = sort(numnz);
n_movie = 1000;
perm = randperm(n_movie);
M_true = M_true(idx(perm),idx(perm));


%% Show SGD works well for well conditon, ScaleSGD workds well for both well and ill condition (n^2 samples)

% Sample Ground Truth
r = 10;
n = size(M_true,1);
percent = 10;
M = samplepair(M_true,percent);

epochs = 500;
lossfun = 'ranklog';
metric = 'AUC';
% metric = 'none';
mo = 0;
mb = 1;

lr1 = 0.1;
lr2 = 1000;
reg = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);  
[~, fsgd, ~, asgd] = psd_sgd(M, r, epochs, lr1, lossfun, mo, mb, reg, X0, M_true, metric);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);  
[~, fscsgd, ~, ascsgd] = psd_scalesgd(M, r, epochs, lr2, lossfun, mo, mb, reg, X0, M_true, metric);fprintf('\n')

% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = inf;
plotfig1(fscsgd,fsgd,ascsgd,asgd,xlimit,percent);

%% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = 10;
plotfig1(fscsgd,fsgd,ascsgd,asgd,xlimit,percent);