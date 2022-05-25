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

%% Euclidean Distance Matrix
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

%% Collaborative Filtering (Jester)
% Generate ground truth item-item matrix using user-item from jester
% dataset (https://eigentaste.berkeley.edu/dataset/)

clear;
data = readmatrix('Data/jester-data-3.xls');
num_rate = data(:,1);
UserItem = data(:,2:end);
UserItem(UserItem==99) = 0;
n = size(UserItem,2);
% Calculate the item-item matrix
ItemItem = zeros(n,n);
for i = 1:n
    for j = i:n
        ItemItem(i,j) = (UserItem(:,i)'*UserItem(:,j))/(norm(UserItem(:,i),2)*norm(UserItem(:,j),2));
        ItemItem(j,i) = ItemItem(i,j);
    end
end
r = rank(ItemItem);
filename = 'JESTER.mat';
save(fullfile('Data',filename),'data','UserItem','ItemItem','r')

%% Collaborative Filtering (MovieLens)
% Generate ground truth item-item matrix using user-item from movielens
% dataset (https://grouplens.org/datasets/movielens/)

clear;
data=readmatrix('Data/ratings.csv');
n_user = max(data(:,1));
n_movie = max(data(:,2));

% Forming the user-item matrix
UserItem = sparse(data(:,1),data(:,2),data(:,3),n_user,n_movie);
% UserItem = zeros(n_user,n_movie);
% for i=1:size(data,1)
%    UserItem(data(i,1),data(i,2)) = data(i,3); 
% end
UserItem = UserItem(:,any(UserItem));
n_user = size(UserItem,1);
n_movie = size(UserItem,2);

% Calculate the item-item matrix
ItemItem = zeros(n_movie,n_movie);
for i = 1:n_movie
    if mod(i,1000)==0
       disp(i) 
    end
    for j = i:n_movie
        ItemItem(i,j) = (UserItem(:,i)'*UserItem(:,j))/(norm(UserItem(:,i),2)*norm(UserItem(:,j),2));
        ItemItem(j,i) = ItemItem(i,j);
    end
end
r = rank(ItemItem);
filename = 'MOVIELENS.mat';
save(fullfile('Data',filename),'data','UserItem','ItemItem','r')

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

%% Synthetic data (n x n Asymmetric Matrix with Rank r)
rng(1)   % Random seed
n = 30;  % Size of matrix
r = 3;   % Rank
% Generate well-conditioned n x n asymmetric matrix with rank r
U = orth(randn(n,n));
V = orth(randn(n,n));
s = [2*ones(1,r),zeros(1,n-r)];
M = U*diag(s)*V';
filename = ['ASYN_Well', num2str(n),'.mat'];
save(fullfile('Data',filename),'M','r')

% Generate ill-conditioned n x n asymmetric matrix with rank r
s = [10.^(-2*(0:r-1)+1),zeros(1,n-r)];
M = U*diag(s)*V';
filename = ['ASYN_ILL', num2str(n),'.mat'];
save(fullfile('Data',filename),'M','r')















