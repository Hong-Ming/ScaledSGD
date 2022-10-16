%% Small
load('Data/CF_Small.mat')
plotCF(fsgd,fscsgd,aucsgd,aucscsgd,np_maximum)

%% Mediu,
load('Data/CF_Medium.mat')
plotCF(fsgd,fscsgd,aucsgd,aucscsgd,np_maximum)

%% Large
load('Data/CF_Large.mat')
plotCF(fsgd,fscsgd,aucsgd,aucscsgd,np_maximum)

%% Large
addpath Functions
load('Data/CF_Huge.mat')
plotCFHuge(fsgd,fscsgd,aucsgd,aucscsgd,np_maximum)