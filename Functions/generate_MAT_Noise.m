function [MW, MI] = generate_MAT_Noise(n, r, SNR)
% Generate well-conditioned and ill-conditioned symmetric n x n matrices
% with rank r, each matrix is corrupted by a white Gaussian noise with
% signal to noise ratio SNR

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

end