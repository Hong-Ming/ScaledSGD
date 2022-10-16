%% Synthetic data (n x n Symmetric Matrix with Rank r)
clear;
rng(1)   % Random seed
n = 30;  % Size of matrix
r = 3;   % Rank

% Generate well-conditioned n x n symmetric matrix with rank r
U = orth(randn(n,n));
s = [2*ones(1,r),zeros(1,n-r)];
MW = U*diag(s)*U';

% Generate ill-conditioned n x n symmetric matrix with rank r
s = [10.^(-2*(0:r-1)+1),zeros(1,n-r)];
MI = U*diag(s)*U';

filename = ['MAT_', num2str(n),'.mat'];
save(fullfile('Data',filename),'MW','MI','r')

%% Synthetic data noisy case (n x n Symmetric Matrix with Rank r)
clear;
rng(1)    % Random seed
n = 30;   % Size of matrix
r = 3;    % Rank
SNR = 15; % Signal to noise ratio

% Generate well-conditioned n x n symmetric matrix with rank r
U = orth(randn(n,n));
s = [10*ones(1,r),zeros(1,n-r)];
MW = U*diag(s)*U';
% Generate noise
rng(1)
sigma = mean(MW(:).^2)/10^(SNR/10);
noise = sqrt(sigma)*randn(n);
noise = (noise+noise').*((1/2-1/sqrt(2))*eye(n)+1/sqrt(2));
MW = MW + noise;

% Generate ill-conditioned n x n symmetric matrix with rank r
s = [10.^(-2*(0:r-1)+1),zeros(1,n-r)];
MI = U*diag(s)*U';
% Generate noise
rng(1)
sigma = mean(MI(:).^2)/10^(SNR/10);
noise = sqrt(sigma)*randn(n);
noise = (noise+noise').*((1/2-1/sqrt(2))*eye(n)+1/sqrt(2));
MI = MI + noise;
filename = ['MAT_Noise_', num2str(n),'.mat'];
save(fullfile('Data',filename),'MW','MI','r','SNR')

%% Euclidean Distance Matrix Completion (EDM)
% Generate n x n Euclidean distance matrix D where D(i,j) = |xi-xj|^2 and 
% x1,...,xn are points in 3 dimensional space

clear;
rng(1)    % Random seed
n = 30;   % Number of points
r = 3;    % Rank

% Experiment 1: Well-conditioned case
% Uniformly sample n points in a cube center at origin with side length 2,
% the coordinates in each points has precision up to four digits. The n
% sample points are store in the rows of X
digits = 4;
precision = 10^digits;
p = 2*precision + 1;
idx = randperm(p^3,n);
[i,j,k] = ind2sub([p,p,p], idx);
XW = ([i',j',k']-precision-1)/precision;

% Calculate Euclidean distance matrix D(i,j) = |xi-xj|^2
DW = zeros(n);
for ii = 1:n
    for jj = ii+1:n
         DW(ii,jj) = norm(XW(ii,:)-XW(jj,:),2)^2+0.0*randn;
         DW(jj,ii) = DW(ii,jj);
    end
end
% Grammian of X
MW = XW*XW';

% Experiment 2: Ill-conditioned case
% Shift the x-coordinates of the first 5 sample point to create a Ill-conditioned X
XI = XW;
XI(1:5,1) = XI(1:5,1)+10;
% Calculate Euclidean distance matrix D(i,j) = |xi-xj|^2
DI = zeros(n);
for ii = 1:n
    for jj = ii+1:n
         DI(ii,jj) = norm(XI(ii,:)-XI(jj,:),2)^2+0.0*randn;
         DI(jj,ii) = DI(ii,jj);
    end
end
% Grammian of X
MI = XI*XI';
filename = ['EDM_', num2str(n),'.mat'];
save(fullfile('Data',filename),'DW','MW','XW','DI','MI','XI','r')

%% Collaborative Filtering (MovieLens Latest Small-scale, 1M sample)
% Generate ground truth item-item matrix using user-item from movielens
% MovieLens Latest Small-scale dataset (https://grouplens.org/datasets/movielens/latest/)
clear;
rng(1)    % Random seed
data=readmatrix('Data/ratings_sm.csv');

million = 1e6;
train_size = 1*million;
test_size = 0.1*million;
m = train_size + test_size;

% Form the user-item matrix
n_user = max(data(:,1)); n_movie = max(data(:,2));
UserItem = sparse(data(:,1),data(:,2),data(:,3),n_user,n_movie);

% Delete items no users rating and update number of movies
ItemKeep = full(sum(logical(UserItem))) > 0;
UserItem = UserItem(:,ItemKeep);
ItemNorm = full(sqrt(sum(UserItem.^2,1)));
n_movie = numel(ItemNorm);

% Generate dense item-item matrix
ItemItem = full(UserItem'*UserItem);
ItemItem = ItemItem./ItemNorm;
ItemItem = ItemItem./ItemNorm';

% Generate random item-item content
sample = 3*m;
[i,j,k] = ind2sub(n_movie*[1,1,1], randperm(n_movie^3, sample));
idx_ij = sub2ind(n_movie*[1,1], i, j);
idx_ik = sub2ind(n_movie*[1,1], i, k);
Mij = ItemItem(idx_ij); Mik = ItemItem(idx_ik);
Yijk = sign(Mij - Mik);

% Strip the comparisons with Mij = Mik since these do not affect training
keep = find(Yijk); keep = keep(1:m);
i = i(keep); j = j(keep); k = k(keep); Yijk = (Yijk(keep)+1)/2;

% Split into train and test set
perm = randperm(m);
i_train = i(perm(test_size+1:end)); j_train = j(perm(test_size+1:end));
k_train = k(perm(test_size+1:end)); Yijk_train = Yijk(perm(test_size+1:end));
i_test = i(perm(1:test_size)); j_test = j(perm(1:test_size));
k_test = k(perm(1:test_size)); Yijk_test = Yijk(perm(1:test_size));

% Output sparse data
spdata.train = [i_train(:),j_train(:),k_train(:),Yijk_train(:)];
spdata.test = [i_test(:),j_test(:),k_test(:),Yijk_test(:)];

filename = 'CF_1M.mat';
save(fullfile('Data',filename), 'spdata', 'n_movie')

%% Collaborative Filtering (MovieLens Latest Large-scale, 10M sample)
% Generate ground truth item-item matrix using user-item from movielens
% MovieLens Latest Large-scale dataset (https://grouplens.org/datasets/movielens/latest/)
clear;
rng(1)    % Random seed
data=readmatrix('Data/ratings_lr.csv');
million = 1e6;
train_size = 10*million;
test_size = 1*million;
m = train_size + test_size;

% Form the user-item matrix
n_user = max(data(:,1));
n_movie = max(data(:,2));
UserItem = sparse(data(:,1),data(:,2),data(:,3),n_user,n_movie);
clear data

% Delete items with no users rating and update number of movies
ItemKeep = full(sum(logical(UserItem))) > 0;
UserItem = UserItem(:,ItemKeep);
ItemNorm = full(sqrt(sum(UserItem.^2,1)));
n_movie = numel(ItemNorm);
clear ItemKeep

% Generate random item-item content
sample = 3*m; worker = 10; batch = sample/worker;
[i,j,k] = ind2sub(n_movie*[1,1,1], randperm(n_movie^3, sample));
i = reshape(i,batch,worker); j = reshape(j,batch,worker); k = reshape(k,batch,worker);
Mij = zeros(batch,worker); Mik = zeros(batch,worker);
parfor pdx = 1:worker
    w = 0;
    ipar = i(:,pdx); jpar = j(:,pdx); kpar = k(:,pdx);
    UI = UserItem; IN = ItemNorm;
    Mijpar = zeros(batch,1); Mikpar = zeros(batch,1);
    for idx = 1:batch
        ii = ipar(idx); jj = jpar(idx); kk = kpar(idx);
        Mijpar(idx) = UI(:,ii)'*UI(:,jj)/(IN(ii)*IN(jj));
        Mikpar(idx) = UI(:,ii)'*UI(:,kk)/(IN(ii)*IN(kk));
        if pdx == worker && mod(idx*worker,million)==0
            w = fprintf([repmat('\b',1,w),'sample: %dM/%dM\n'],idx*worker/million,sample/million) - w; 
        end
    end
    Mij(:,pdx) = Mijpar; Mik(:,pdx) = Mikpar;
end

i = i(:)'; j = j(:)'; k = k(:)'; Mij = Mij(:)'; Mik = Mik(:)';
Yijk = sign(Mij - Mik);
clear UserItem ItemNorm Mij Mik

% Strip the comparisons with Mij = Mik since these do not affect training
keep = find(Yijk); keep = keep(1:m);
i = i(keep); j = j(keep); k = k(keep); Yijk = (Yijk(keep)+1)/2;
clear keep

% Split into train and test set
perm = randperm(m);
i_train = i(perm(test_size+1:end)); j_train = j(perm(test_size+1:end));
k_train = k(perm(test_size+1:end)); Yijk_train = Yijk(perm(test_size+1:end));
i_test = i(perm(1:test_size)); j_test = j(perm(1:test_size));
k_test = k(perm(1:test_size)); Yijk_test = Yijk(perm(1:test_size));
clear i j k Yijk perm

% Output sparse data
spdata.train = [i_train(:),j_train(:),k_train(:),Yijk_train(:)];
spdata.test = [i_test(:),j_test(:),k_test(:),Yijk_test(:)];

filename = 'CF_10M.mat';
save(fullfile('Data',filename), 'spdata', 'n_movie')

%% Collaborative Filtering (MovieLens Latest Large-scale, 30M sample)
% Generate ground truth item-item matrix using user-item from movielens
% MovieLens Latest Large-scale dataset (https://grouplens.org/datasets/movielens/latest/)
clear;
rng(1)    % Random seed
data=readmatrix('Data/ratings_lr.csv');
million = 1e6;
train_size = 30*million;
test_size = 3*million;
m = train_size + test_size;

% Form the user-item matrix
n_user = max(data(:,1));
n_movie = max(data(:,2));
UserItem = sparse(data(:,1),data(:,2),data(:,3),n_user,n_movie);
clear data

% Delete items with no users rating and update number of movies
ItemKeep = full(sum(logical(UserItem))) > 0;
UserItem = UserItem(:,ItemKeep);
ItemNorm = full(sqrt(sum(UserItem.^2,1)));
n_movie = numel(ItemNorm);
clear ItemKeep

% Generate random item-item content
sample = 3*m; worker = 10; batch = sample/worker;
[i,j,k] = ind2sub(n_movie*[1,1,1], randperm(n_movie^3, sample));
i = reshape(i,batch,worker); j = reshape(j,batch,worker); k = reshape(k,batch,worker);
Mij = zeros(batch,worker); Mik = zeros(batch,worker);
parfor pdx = 1:worker
    w = 0;
    ipar = i(:,pdx); jpar = j(:,pdx); kpar = k(:,pdx);
    UI = UserItem; IN = ItemNorm;
    Mijpar = zeros(batch,1); Mikpar = zeros(batch,1);
    for idx = 1:batch
        ii = ipar(idx); jj = jpar(idx); kk = kpar(idx);
        Mijpar(idx) = UI(:,ii)'*UI(:,jj)/(IN(ii)*IN(jj));
        Mikpar(idx) = UI(:,ii)'*UI(:,kk)/(IN(ii)*IN(kk));
        if pdx == worker && mod(idx*worker,million)==0
            w = fprintf([repmat('\b',1,w),'sample: %dM/%dM\n'],idx*worker/million,sample/million) - w; 
        end
    end
    Mij(:,pdx) = Mijpar; Mik(:,pdx) = Mikpar;
end

i = i(:)'; j = j(:)'; k = k(:)'; Mij = Mij(:)'; Mik = Mik(:)';
Yijk = sign(Mij - Mik);
clear UserItem ItemNorm Mij Mik

% Strip the comparisons with Mij = Mik since these do not affect training
keep = find(Yijk); keep = keep(1:m);
i = i(keep); j = j(keep); k = k(keep); Yijk = (Yijk(keep)+1)/2;
clear keep

% Split into train and test set
perm = randperm(m);
i_train = i(perm(test_size+1:end)); j_train = j(perm(test_size+1:end));
k_train = k(perm(test_size+1:end)); Yijk_train = Yijk(perm(test_size+1:end));
i_test = i(perm(1:test_size)); j_test = j(perm(1:test_size));
k_test = k(perm(1:test_size)); Yijk_test = Yijk(perm(1:test_size));
clear i j k Yijk perm

% Output sparse data
spdata.train = [i_train(:),j_train(:),k_train(:),Yijk_train(:)];
spdata.test = [i_test(:),j_test(:),k_test(:),Yijk_test(:)];

filename = 'CF_30M.mat';
save(fullfile('Data',filename), 'spdata', 'n_movie')


%% Collaborative Filtering (MovieLens25M, 100M sample)
% Generate ground truth item-item matrix using user-item from MovieLens25M
% MovieLens25M dataset (https://grouplens.org/datasets/movielens/25m/)
clear;
rng(1)    % Random seed
data=readmatrix('Data/ratings25M.csv');
million = 1e6;
train_size = 100*million;
test_size = 10*million;
m = train_size + test_size;

% Form the user-item matrix
n_user = max(data(:,1));
n_movie = max(data(:,2));
UserItem = sparse(data(:,1),data(:,2),data(:,3),n_user,n_movie);
clear data

% Delete items no users rating and update number of movies
ItemKeep = full(sum(logical(UserItem))) > 0;
UserItem = UserItem(:,ItemKeep);
ItemNorm = full(sqrt(sum(UserItem.^2,1)));
n_movie = numel(ItemNorm);
clear ItemKeep

% Generate random item-item content
sample = 3*m; worker = 10; batch = sample/worker;
[i,j,k] = ind2sub(n_movie*[1,1,1], randperm(n_movie^3, sample));
i = reshape(i,batch,worker); j = reshape(j,batch,worker); k = reshape(k,batch,worker);
Mij = zeros(batch,worker); Mik = zeros(batch,worker);
parfor pdx = 1:worker
    w = 0;
    ipar = i(:,pdx); jpar = j(:,pdx); kpar = k(:,pdx);
    UI = UserItem; IN = ItemNorm;
    Mijpar = zeros(batch,1); Mikpar = zeros(batch,1);
    for idx = 1:batch
        ii = ipar(idx); jj = jpar(idx); kk = kpar(idx);
        Mijpar(idx) = UI(:,ii)'*UI(:,jj)/(IN(ii)*IN(jj));
        Mikpar(idx) = UI(:,ii)'*UI(:,kk)/(IN(ii)*IN(kk));
        if pdx == worker && mod(idx*worker,million)==0
            w = fprintf([repmat('\b',1,w),'sample: %dM/%dM\n'],idx*worker/million,sample/million) - w; 
        end
    end
    Mij(:,pdx) = Mijpar; Mik(:,pdx) = Mikpar;
end

i = i(:)'; j = j(:)'; k = k(:)'; Mij = Mij(:)'; Mik = Mik(:)';
Yijk = sign(Mij - Mik);
clear UserItem ItemNorm Mij Mik

% Strip the comparisons with Mij = Mik since these do not affect training
keep = find(Yijk); keep = keep(1:m);
i = i(keep); j = j(keep); k = k(keep); Yijk = (Yijk(keep)+1)/2;
clear keep

% Split into train and test set
perm = randperm(m);
i_train = i(perm(test_size+1:end)); j_train = j(perm(test_size+1:end));
k_train = k(perm(test_size+1:end)); Yijk_train = Yijk(perm(test_size+1:end));
i_test = i(perm(1:test_size)); j_test = j(perm(1:test_size));
k_test = k(perm(1:test_size)); Yijk_test = Yijk(perm(1:test_size));
clear i j k Yijk perm

% Output sparse data
spdata.train = [i_train(:),j_train(:),k_train(:),Yijk_train(:)];
spdata.test = [i_test(:),j_test(:),k_test(:),Yijk_test(:)];

filename = 'CF_100M.mat';
save(fullfile('Data',filename), 'spdata', 'n_movie', '-v7.3')

