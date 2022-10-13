%% Collaborative Filtering (MovieLens500k)
clear
addpath Functions
loader = load('Data/MovieLens_small.mat');
spdata = loader.spdata;
most_popular = loader.most_popular;
d = loader.n_movie;
r = 10;
epochs = 50;

%% SGD
lr_sgd = 3e-2;
rng(1); fprintf('\n')
[~,fsgd,aucsgd] = bpr_scaledsgd(spdata, d, r, epochs, lr_sgd, false);

%% PrecSGD
lr_scaledsgd = 3e2;
rng(1); fprintf('\n')
[~,fscsgd,aucscsgd] = bpr_scaledsgd(spdata, d, r, epochs, lr_scaledsgd, true);

%% Non-personalized Max
lr_np = 1e-1; 
rng(1); fprintf('\n')
[np_max] = bpr_np(spdata, d, epochs, lr_np);

save('TaskSmall500k.mat','fsgd','fscsgd','aucsgd','aucscsgd','most_popular','np_max')

%%
xlimit = 50;
load('TaskSmall500k')
plotfig2(fsgd,fscsgd,aucsgd,aucscsgd,most_popular,np_max,xlimit)




