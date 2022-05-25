%% Load Data (Square Loss Noisy Setting)
clc;clear
addpath Functions
% 30 x 30 symmetric well condtioned matrix, rank = 3
loader = load('Data/SYN_WELL_Noise30.mat'); 
MW_true = loader.M;

% 30 x 30 symmetric ill condtioned matrix, rank = 3
loader = load('Data/SYN_ILL_Noise30.mat'); 
MI_true = loader.M;

n = 30;
r = 10;

[U,S] = eig(MW_true); [~,idx] = sort(diag(S),'descend'); perm = idx(1:r);
U = U(:,perm); S = S(perm,perm); 
nfw = (1/2)*(1/n^2)*norm(MW_true-U*S*U','fro')^2;

[U,S] = eig(MI_true); [~,idx] = sort(diag(S),'descend'); perm = idx(1:r);
U = U(:,perm); S = S(perm,perm); 
nfi = (1/2)*(1/n^2)*norm(MI_true-U*S*U','fro')^2;




%% Show SGD works well for well conditon, ScaleSGD workds well for both well and ill condition (n^2 samples)

% Sample Ground Truth
percent = 100;
MW = sampling(MW_true,percent);
MI = sampling(MI_true,percent);

epochs = 200;
lr = 0.01; 
lossfun = 'square';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fsgdwell, ~] = psd_sgd(MW, r, epochs, lr, lossfun, [], [], [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fsgdill, ~] = psd_sgd(MI, r, epochs, lr, lossfun, [], [], [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fscsgdwell, ~] = psd_scalesgd(MW, r, epochs, 15*lr, lossfun, [], [], [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);
[~, fscsgdill, ~] = psd_scalesgd(MI, r, epochs, 15*lr, lossfun, [], [], [], X0);fprintf('\n')

% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = 50;
plotfig2(fscsgdwell,fsgdwell,fscsgdill,fsgdill,nfw,nfi,xlimit);

%% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = 30;
plotfig2(fscsgdwell,fsgdwell,fscsgdill,fsgdill,nfw,nfi,xlimit);