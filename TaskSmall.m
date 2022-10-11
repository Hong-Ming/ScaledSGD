%% Collaborative Filtering (MovieLens500k)
clear
loader = load('Data/MovieLens_small.mat');
spdata = loader.spdata;
d = loader.n_movie;
r = 10;
epochs = 50;

learning_rate_sgd = 1e-1;
learning_rate_scaledsgd = 1e2;

rng(1); fprintf('\n')
[~,fsgd,~,esgd,esgdt] = bpr_scalesgd(spdata, d, r, epochs, learning_rate_sgd, false);

rng(1); fprintf('\n')
[~,fscsgd,~,escsgd,escsgdt] = bpr_scalesgd(spdata, d, r, epochs, learning_rate_scaledsgd, true);

save('TaskSmall500k.mat','fsgd','fscsgd','esgd','escsgd','esgdt','escsgdt')

%%
xlimit = 50;
load('TaskSmall500k')
plotfig2(fsgd,fscsgd,esgd,escsgd,esgdt,escsgdt,xlimit)




