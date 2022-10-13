%% MovieLens 500k
xlimit = 20;
load('TaskSmall500k')
plotfig2(fsgd,fscsgd,esgd,escsgd,esgdt,escsgdt,most_popular,xlimit)

%% MovieLens 2.7M
xlimit = 20;
load('TaskLarge2.7M.mat')
plotfig2(fsgd,fscsgd,esgd,escsgd,esgdt,escsgdt,most_popular,xlimit)

%% MovieLens 10M
xlimit = 5;
load('TaskLarge10M.mat')
plotfig2(fsgd,fscsgd,esgd,escsgd,esgdt,escsgdt,most_popular,xlimit)