function spmat= samplepair(M,percent)
n = size(M,1);
n1 = size(M,2);
t_row = sum(M~=0,2);              % vector contains nnz of each row of M
t_pair = sum(t_row.*(t_row-1))/2; % number of pairwise sample of M
n_row = zeros(size(M,1),1);       % vector contains nnz of each row of spmat
n_pair = sum(n_row.*(n_row-1))/2; % number of pairwise sample of spmat
 
% Greedy way to select number of nonzero elemt in each row, the goal is to
% increase entries in n_row to make n_pair >= t_pair*(percent/100)
i = 1;
perm = randperm(numel(n_row));
while n_pair < t_pair*percent/100
    idx = perm(i);
    n_add = randi([1,10]);
    n_row(idx) = min(n_row(idx)+n_add, t_row(idx));
    n_pair = sum(n_row.*(n_row-1))/2;
    if i == numel(n_row)
        i = 1;
        perm = randperm(numel(n_row));
    else
        i = i + 1;
    end
end

% Sample each row based on n_row
i = cell(1,n); j = cell(1,n); k = cell(1,n);
for row = 1:n
    thisrow = M(row,:);
    rr = find(thisrow);     % Find indices of nonzero elemt
    idx = randperm(numel(rr),n_row(row));
    i{row} = row*ones(1,n_row(row));
    j{row} = rr(idx);
    k{row} = thisrow(rr(idx));
end
i = cell2mat(i); j = cell2mat(j); k = cell2mat(k);
m = sum(n_row);
spmat = sparse(i,j,k,n,n1,m);
    
end