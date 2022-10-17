function [MW, MI] = generate_MAT(n,r)
% Generate well-conditioned and ill-conditioned symmetric n x n matrices with rank r

% Generate well-conditioned n x n symmetric matrix with rank r
U = orth(randn(n,n));
s = [2*ones(1,r),zeros(1,n-r)];
MW = U*diag(s)*U';

% Generate ill-conditioned n x n symmetric matrix with rank r
s = [10.^(-2*(0:r-1)+1),zeros(1,n-r)];
MI = U*diag(s)*U';

end