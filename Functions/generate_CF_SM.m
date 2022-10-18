function [spdata, n_movie] = generate_CF_SM(data, train_size, test_size)
% Generate samples of ground truth item-item matrix from MovieLens dataset
% This code is used for small-scale dataset for which the entire item-item matrix
% can be constructed from user-item matrix. For large-scale dataset, the
% item-item matrix might be too large to fit into the memory, please use the
% function 'generate_CF_LR' in instead.

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
% Note that we sample 3 times more because some of the samples might have
% Mij = Mik, which we will delete it later.
sample = 3*m;
[i,j,k] = ind2sub(n_movie*[1,1,1], randperm(n_movie^3, sample));
idx_ij = sub2ind(n_movie*[1,1], i, j);
idx_ik = sub2ind(n_movie*[1,1], i, k);
Mij = ItemItem(idx_ij); Mik = ItemItem(idx_ik);
Yijk = sign(Mij - Mik);

% Keep only the first m samples that Mij ~= Mik
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


end