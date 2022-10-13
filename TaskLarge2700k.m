%% Collaborative Filtering (MovieLens2.7M)
clear
addpath Functions
loader = load('Data/MovieLens2.7M.mat');
spdata = loader.spdata;
most_popular = loader.most_popular;
d = loader.n_movie;
r = 4;
epochs = 25;

%% SGD
lr_sgd = 3e-2;
rng(1); fprintf('\n')
[~,fsgd,~,esgd,esgdt] = bpr_scaledsgd(spdata, d, r, epochs, lr_sgd, false);

%% PrecSGD
lr_scaledsgd = 3e2;
rng(1); fprintf('\n')
[~,fscsgd,~,escsgd,escsgdt] = bpr_scaledsgd(spdata, d, r, epochs, lr_scaledsgd, true);

%% Non-personalized Max
lr_np = 1e-1; 
rng(1); fprintf('\n')
[np_max] = bpr_np(spdata, d, epochs, lr_np);

save('TaskLarge2.7M.mat','fsgd','fscsgd','esgd','escsgd','esgdt','escsgdt','most_popular','np_max')

%%
xlimit = 10;
load('TaskLarge2.7M.mat')
plotfig2(fsgd,fscsgd,esgd,escsgd,esgdt,escsgdt,most_popular,np_max,xlimit)





