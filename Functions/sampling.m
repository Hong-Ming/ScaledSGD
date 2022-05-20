function spmat = sampling(M,m)
n = size(M,1);
n1 = size(M,2);
idx = randperm(n*n1, m);
[i,j] = ind2sub([n,n1], idx);
k = M(idx);
k(k==0)=1e-15;
spmat = sparse(i,j,k,n,n1);
end