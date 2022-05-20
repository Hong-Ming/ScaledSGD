%% Synthetic data (n x n Symmetric Matrix with Rank r)
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

%% Collaborative Filtering
% Generate ground truth item-item matrix using user-item from jester
% dataset (https://eigentaste.berkeley.edu/dataset/)
data = readmatrix('Data/jester-data-3.xls');
num_rate = data(:,1);
UserItem = data(:,2:end);
UserItem(UserItem==99) = 0;
n = size(UserItem,2);
% Calculate the item-item matrix M
M = zeros(n,n);
for i = 1:n
    for j = i:n
        M(i,j) = (UserItem(:,i)'*UserItem(:,j))/(norm(UserItem(:,i),2)*norm(UserItem(:,j),2));
        M(j,i) = M(i,j);
    end
end
r = rank(M);
filename = 'JESTER.mat';
save(fullfile('Data',filename),'data','UserItem','M','r')

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















