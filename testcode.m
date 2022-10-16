clear
addpath Functions

loader = load('Data/CF_1M.mat');
spdata = loader.spdata;
d = loader.n_movie;
r = 3;
epochs = 5;

rng(1); learning_rate = 1e-1; 
[~, np_maximum] = bpr_npmaximum(spdata, d, 100, learning_rate);