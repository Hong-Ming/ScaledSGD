%% Load Data (Square Loss)
clear
addpath Functions

% 30 x 30 symmetric well condtioned matrix, rank = 3
loader = load('Data/SYN_WELL30.mat'); 
MW_true = loader.M;

% 30 x 30 symmetric ill condtioned matrix, rank = 3
loader = load('Data/SYN_ILL30.mat'); 
MI_true = loader.M;

% Sample Ground Truth
percent = 100;
MW = sampling(MW_true,percent);
MI = sampling(MI_true,percent);
r = 3;
epochs = 500;
learning_rate = 0.3; 
lossfun = 'square';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fsgdwell] = sgd(MW, r, epochs, learning_rate, lossfun);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fsgdill] = sgd(MI, r, epochs, learning_rate, lossfun);

%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well ScaledSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fscsgdwell] = scaledsgd(MW, r, epochs, learning_rate, lossfun);

%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill ScaledSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fscsgdill] = scaledsgd(MI, r, epochs, learning_rate, lossfun);

% Plot ScaledSGD vs SGD (Well condition and Ill condition)
xlimit = 200;
plotfig(fscsgdwell,fsgdwell,fscsgdill,fsgdill,xlimit);

