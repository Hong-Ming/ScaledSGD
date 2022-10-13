%% Collaborative Filtering (MovieLens2.7M)
clear
addpath Functions
loader = load('Data/MovieLens10M.mat');
spdata = loader.spdata;
most_popular = loader.most_popular;
d = loader.n_movie;
r = 4;
epochs = 50;
lr_pgd = 1e-1;
lr_scaledpgd = 1e2;

rng(1); fprintf('\n')
[~,fsgd,~,esgd,esgdt] = bpr_scaledsgd(spdata, d, r, epochs, lr_pgd, false);

rng(1); fprintf('\n')
[~,fscsgd,~,escsgd,escsgdt] = bpr_scaledsgd(spdata, d, r, epochs, lr_scaledpgd, true);

save('TaskLarge10M.mat','fsgd','fscsgd','esgd','escsgd','esgdt','escsgdt','most_popular')

%%
xlimit = 50;
load('TaskLarge10M.mat')
plotfig2(fsgd,fscsgd,esgd,escsgd,esgdt,escsgdt,most_popular,xlimit)






