%% Collaborative Filtering (MovieLens)
% Generate ground truth item-item matrix using user-item from movielens
% dataset (https://grouplens.org/datasets/movielens/)
clear;
data=readmatrix('Data/ratings_sm.csv');
n_user = max(data(:,1));
n_movie = max(data(:,2));
m = 5e5;

% Form the user-item matrix
UserItem = sparse(data(:,1),data(:,2),data(:,3),n_user,n_movie);
ItemCount = full(sum(logical(UserItem))); 

% Delete items with too few users and update number of movies
ItemKeep = ItemCount>3;
UserItem = UserItem(:,ItemKeep);
ItemNorm = full(sqrt(sum(UserItem).^2));
n_movie = numel(ItemNorm);

% Generate dense item-item matrix
ItemItem = full(UserItem'*UserItem);
ItemItem = ItemItem./ItemNorm;
ItemItem = ItemItem./ItemNorm';

% Generate random item-item content
idx = randperm(n_movie^3, m);
[i,j,k] = ind2sub(n_movie*[1,1,1], idx);
idx_ij = sub2ind(n_movie*[1,1], i, j);
idx_ik = sub2ind(n_movie*[1,1], i, k);
Mij = ItemItem(idx_ij); Mik = ItemItem(idx_ik);
Yijk = sign(Mij - Mik);

% Strip the comparisons with Mij = Mik since these do not affect training
keep = Yijk ~= 0;
i = i(keep); j = j(keep); k = k(keep); Yijk = Yijk(keep);

% Output sparse data
spdata = [i(:),j(:),k(:),(Yijk(:)+1)/2];
%save TaskSmallData spdata n_movie

%%
clear
load TaskSmallData;
%%
rng(0);
[X,ftrain1,ftest1,etrain1,etest1,gradnrm1] = ...
    bpr_scalesgd(spdata, n_movie, 10, 50, 2e-2, false);

rng(0);
[X,ftrain2,ftest2,etrain2,etest2,gradnrm2] = ...
    bpr_scalesgd(spdata, n_movie, 10, 50, 2e1, true);

figure(1); clf;
hold all
plot(ftrain1);
plot(ftrain2);

figure(2); clf;
hold all
plot(etrain1);
plot(etrain2);

figure(3); clf;
hold all
plot(etest1);
plot(etest2);




