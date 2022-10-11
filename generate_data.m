%% Synthetic data (n x n Symmetric Matrix with Rank r)
clear;
rng(1)   % Random seed
n = 30;  % Size of matrix
r = 3;   % Rank
% Generate well-conditioned n x n symmetric matrix with rank r
U = orth(randn(n,n));
s = [2*ones(1,r),zeros(1,n-r)];
M = U*diag(s)*U';
filename = ['SYN_WELL', num2str(n),'.mat'];
save(fullfile('Data',filename),'M','r')

% Generate ill-conditioned n x n symmetric matrix with rank r
s = [10.^(-2*(0:r-1)+1),zeros(1,n-r)];
M = U*diag(s)*U';
filename = ['SYN_ILL', num2str(n),'.mat'];
save(fullfile('Data',filename),'M','r')

%% Synthetic data noisy case (n x n Symmetric Matrix with Rank r)
clear;
rng(1)    % Random seed
n = 30;   % Size of matrix
r = 3;    % Rank
SNR = 15; % Signal to noise ratio

% Generate well-conditioned n x n symmetric matrix with rank r
U = orth(randn(n,n));
s = [10*ones(1,r),zeros(1,n-r)];
M = U*diag(s)*U';
% Generate noise
rng(1)
sigma = mean(M(:).^2)/10^(SNR/10);
noise = sqrt(sigma)*randn(n);
noise = (noise+noise').*((1/2-1/sqrt(2))*eye(n)+1/sqrt(2));
M = M + noise;
filename = ['SYN_WELL_Noise', num2str(n),'.mat'];
save(fullfile('Data',filename),'M','r', 'SNR')

% Generate ill-conditioned n x n symmetric matrix with rank r
s = [10.^(-2*(0:r-1)+1),zeros(1,n-r)];
M = U*diag(s)*U';
% Generate noise
rng(1)
sigma = mean(M(:).^2)/10^(SNR/10);
noise = sqrt(sigma)*randn(n);
noise = (noise+noise').*((1/2-1/sqrt(2))*eye(n)+1/sqrt(2));
M = M + noise;
filename = ['SYN_ILL_Noise', num2str(n),'.mat'];
save(fullfile('Data',filename),'M','r','SNR')

%% Euclidean Distance Matrix Completion (EDM)
% Generate n x n Euclidean distance matrix D where D(i,j) = |xi-xj|^2 and 
% x1,...,xn are points in 3 dimensional space

clear;
rng(1)         % Random seed
n = 30;        % Number of points
% Experiment 1: Well-conditioned case
% Uniformly sample n points in a cube center at origin with side length 2,
% the coordinates in each points has precision up to four digits. The n
% sample points are store in the rows of X
digits = 4;
precision = 10^digits;
p = 2*precision + 1;
idx = randperm(p^3,n);
[i,j,k] = ind2sub([p,p,p], idx);
X = ([i',j',k']-precision-1)/precision;
% Calculate Euclidean distance matrix D(i,j) = |xi-xj|^2
D = zeros(n);
for ii = 1:n
    for jj = ii+1:n
         D(ii,jj) = norm(X(ii,:)-X(jj,:),2)^2+0.0*randn;
         D(jj,ii) = D(ii,jj);
    end
end
% Grammian of X
M = X*X';
r = rank(M);
filename = ['EDM_WELL', num2str(n),'.mat'];
save(fullfile('Data',filename),'D','M','X','r')

% Experiment 2: Ill-conditioned case
% Shift the x-coordinates of the first 5 sample point to create a
% Ill-conditioned X
X(1:5,1) = X(1:5,1)+10;
% Calculate Euclidean distance matrix D(i,j) = |xi-xj|^2
D = zeros(n);
for ii = 1:n
    for jj = ii+1:n
         D(ii,jj) = norm(X(ii,:)-X(jj,:),2)^2+0.0*randn;
         D(jj,ii) = D(ii,jj);
    end
end
% Grammian of X
M = X*X';
r = rank(M);
filename = ['EDM_ILL', num2str(n),'.mat'];
save(fullfile('Data',filename),'D','M','X','r')

%% Collaborative Filtering (MovieLens 500k)
% Generate ground truth item-item matrix using user-item from movielens
% dataset (https://grouplens.org/datasets/movielens/)
clear;
rng(1)    % Random seed
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
ItemNorm = full(sqrt(sum(UserItem.^2,1)));
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
Yijk = (Yijk+1)/2;

% Compute nonpersonalize lowerbound on AUC
np_max = sparse([i,i],[j,k],[Mij(keep),Mik(keep)],n_movie,n_movie);
np_max = sum(np_max,1);
% thelower = thelower(:);
np_max = np_max(j)-np_max(k);
np_max = ((np_max > 0) & Yijk) | ((np_max < 0) & ~Yijk);
np_max = mean(np_max);

% Output sparse data
spdata = [i(:),j(:),k(:),Yijk(:)];

filename = 'MovieLens_small.mat';
save(fullfile('Data',filename), 'spdata', 'n_movie', 'np_max')

%% Collaborative Filtering (MovieLens 2.7M)
% Generate ground truth item-item matrix using user-item from movielens
% dataset (https://grouplens.org/datasets/movielens/)
clear;
rng(1)    % Random seed
data=readmatrix('Data/ratings.csv');
n_user = max(data(:,1));
n_movie = max(data(:,2));
m = 2.7e6;

% Form the user-item matrix
UserItem = sparse(data(:,1),data(:,2),data(:,3),n_user,n_movie);
ItemNorm = full(sqrt(sum(UserItem.^2,1)));

% Delete items with no movies and update number of movies
ItemKeep = ItemNorm>0;
UserItem = UserItem(:,ItemKeep);
ItemNorm = ItemNorm(ItemKeep);
n_movie = numel(ItemNorm);

idx = randperm(n_movie^3, m);
[i,j,k] = ind2sub(n_movie*[1,1,1], idx);
Mij = zeros(1,m); Mik = zeros(1,m);
for idx = 1:m
    ii = i(idx); jj = j(idx); kk = k(idx);
    Mij(idx) = UserItem(:,ii)'*UserItem(:,jj)/(ItemNorm(ii)*ItemNorm(jj));
    Mik(idx) = UserItem(:,ii)'*UserItem(:,kk)/(ItemNorm(ii)*ItemNorm(kk));
    if mod(idx,1e4)==0, disp(idx); end
end
Yijk = sign(Mij - Mik);
% Strip the comparisons with Mij = Mik since these do not affect training
keep = Yijk ~= 0;
i = i(keep); j = j(keep); k = k(keep); Yijk = Yijk(keep);
Yijk = (Yijk+1)/2;

% Compute nonpersonalize lowerbound on AUC
np_max = sparse([i,i],[j,k],[Mij(keep),Mik(keep)],n_movie,n_movie);
np_max = sum(np_max,1);
np_max = np_max(j)-np_max(k);
np_max = ((np_max > 0) & Yijk) | ((np_max < 0) & ~Yijk);
np_max = mean(np_max);

% Output sparse data
spdata = [i(:),j(:),k(:),Yijk(:)];
filename = 'MovieLens2.7M.mat';
save(fullfile('Data',filename), 'spdata', 'n_movie', 'np_max')

%% Collaborative Filtering (MovieLens 10M)
% Generate ground truth item-item matrix using user-item from movielens
% dataset (https://grouplens.org/datasets/movielens/)
clear;
rng(1)    % Random seed
data=readmatrix('Data/ratings.csv');
n_user = max(data(:,1));
n_movie = max(data(:,2));
m = 1e7;

% Form the user-item matrix
UserItem = sparse(data(:,1),data(:,2),data(:,3),n_user,n_movie);
ItemNorm = full(sqrt(sum(UserItem.^2,1)));

% Delete items with no movies and update number of movies
ItemKeep = ItemNorm>0;
UserItem = UserItem(:,ItemKeep);
ItemNorm = ItemNorm(ItemKeep);
n_movie = numel(ItemNorm);

idx = randperm(n_movie^3, m);
[i,j,k] = ind2sub(n_movie*[1,1,1], idx);
Mij = zeros(1,m); Mik = zeros(1,m);
for idx = 1:m
    ii = i(idx); jj = j(idx); kk = k(idx);
    Mij(idx) = UserItem(:,ii)'*UserItem(:,jj)/(ItemNorm(ii)*ItemNorm(jj));
    Mik(idx) = UserItem(:,ii)'*UserItem(:,kk)/(ItemNorm(ii)*ItemNorm(kk));
    if mod(idx,1e4)==0, disp(idx); end
end
Yijk = sign(Mij - Mik);
% Strip the comparisons with Mij = Mik since these do not affect training
keep = Yijk ~= 0;
i = i(keep); j = j(keep); k = k(keep); Yijk = Yijk(keep);
Yijk = (Yijk+1)/2;

% Compute nonpersonalize lowerbound on AUC
np_max = sparse([i,i],[j,k],[Mij(keep),Mik(keep)],n_movie,n_movie);
np_max = sum(np_max,1);
np_max = np_max(j)-np_max(k);
np_max = ((np_max > 0) & Yijk) | ((np_max < 0) & ~Yijk);
np_max = mean(np_max);

% Output sparse data
spdata = [i(:),j(:),k(:),Yijk(:)];
filename = 'MovieLens10M.mat';
save(fullfile('Data',filename), 'spdata', 'n_movie')
















