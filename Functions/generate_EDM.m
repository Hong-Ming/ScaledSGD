function [DW, MW, XW, DI, MI, XI] = generate_EDM(n)
% Generate n x n Euclidean distance matrix D where D(i,j) = |xi-xj|^2 and 
% x1,...,xn are points in 3 dimensional space

% Experiment 1: Well-conditioned case
% Uniformly sample n points in a cube center at origin with side length 2,
% the coordinates in each points has precision up to four digits. The n
% sample points are store in the rows of XW
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
% Grammian of XW
MW = XW*XW';

% Experiment 2: Ill-conditioned case
% Shift the x-coordinates of the first 5 sample point to create a
% Ill-conditioned XI
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
% Grammian of XI
MI = XI*XI';

end