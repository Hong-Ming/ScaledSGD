%% Collaborative Filtering (MovieLens)
% Generate ground truth item-item matrix using user-item from movielens
% dataset (https://grouplens.org/datasets/movielens/)
clear;
data=readmatrix('Data/ratings.csv');
n_user = max(data(:,1));
n_movie = max(data(:,2));

% Form the user-item matrix
UserItem = sparse(data(:,1),data(:,2),data(:,3),n_user,n_movie);
ItemCount = sum(logical(UserItem),1);
ItemNorm = full(sqrt(sum(UserItem.^2,1)));

% Delete items with no movies and update number of movies
ItemKeep = ItemNorm>0;
UserItem = UserItem(:,ItemKeep);
ItemNorm = ItemNorm(ItemKeep);
n_movie = numel(ItemNorm);
%%
% Generate random item-item content
m = 2e7;
idx = randperm(n_movie^3, m);
[i,j,k] = ind2sub(n_movie*[1,1,1], idx);
Yijk = zeros(m,1);
for idx = 1:m
    ii = i(idx); jj = j(idx); kk = k(idx);
    Mij = UserItem(:,ii)'*UserItem(:,jj);
    Mik = UserItem(:,ii)'*UserItem(:,kk);
    Yijk(idx) = sign(Mij - Mik);
    if mod(idx,1e4)==0, disp(idx); end
end

% Strip the comparisons with Mij = Mik since these do not affect training
keep = Yijk ~= 0;
i = i(keep); j = j(keep); k = k(keep); Yijk = Yijk(keep);

% Output sparse data
spdata = [i(:),j(:),k(:),(Yijk(:)+1)/2];
save TaskLargeData spdata n_movie

%%
clear
load TaskLargeData
rng(0);
[X,ftrain1,ftest1,etrain1,etest1,gradnrm1] = bpr_scalesgd(spdata, n_movie, 4, 25*2, 1e-1/5, false);

rng(0);
[X,ftrain2,ftest2,etrain2,etest2,gradnrm2] = bpr_scalesgd(spdata, n_movie, 4, 25*2, 1e3/5, true);

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






