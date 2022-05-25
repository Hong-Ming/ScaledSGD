function spmat = sampling(M,percent)
n = size(M,1);
n1 = size(M,2);
[i,j,k] = find(M);
m = floor(numel(i)*percent/100);
idx = randperm(numel(i),m);
i = i(idx);
j = j(idx);
k = k(idx);
spmat = sparse(i,j,k,n,n1,m);
end