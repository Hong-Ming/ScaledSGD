%% Collaborative Filtering (MovieLens2.7M)
clear
loader = load('Data/MovieLens2.7M.mat');
spdata = loader.spdata;
d = loader.n_movie;
r = 4;
epochs = 50;
lr_pgd = 1e-1;
lr_scaledpgd = 1e2;

rng(1); fprintf('\n')
[~,fsgd,~,esgd,esgdt] = bpr_scalesgd(spdata, d, r, epochs, lr_pgd, false);

rng(1); fprintf('\n')
[~,fscsgd,~,escsgd,escsgdt] = bpr_scalesgd(spdata, d, r, epochs, lr_scaledpgd, true);

save('TaskLarge2.7M.mat','fsgd','fscsgd','esgd','escsgd','esgdt','escsgdt')

%%
xlimit = 50;
plotfig2(fsgd,fscsgd,esgd,escsgd,esgdt,escsgdt,xlimit)





