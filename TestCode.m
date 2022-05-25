%% Input Ground Truth (Synthetic Data ILL-conditioned)
clc;clear
addpath Functions

loader = load('Data/SYN_ILL30.mat'); 
M = loader.M; n = size(M,1); r = loader.r;

% Number of sample
m = n^2;
% m = 2*n*r;
spmat = sampling(M,m);
n2 = nnz(spmat);
n3 = sum(spmat~=0,2);
n3 = sum(n3.*(n3-1))/2;

% define colors
Starford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Michigan_Yaize = '#FFCB05';
Rackham_Green  = '#75988d';
Illini_Blue    = '#13294B';
%% Run
epochs = 100; 
momentum = 0;
minibatch = 1;
% lossfun = 'square';
% lossfun = '1bit';
% lossfun = 'pair';
lossfun = 'ranklog';
% lossfun = 'rank';
% lossfun = 'dist';

switch lossfun
    case 'square'
        m = ceil(n2/minibatch);
        alpha = 1; rho = 20;
    case '1bit'
        m = ceil(n2/minibatch);
        alpha = 10; rho = 10;
    case 'dist'
        m = ceil(n2/minibatch);
        alpha = 0.001; rho = 50;
    case 'pair'
        m = ceil(n3/minibatch);
        alpha = 1; rho = 10;
    case 'ranklog'
        m = ceil(n3/minibatch);
        alpha = 1; rho = 100;
    case 'rank'
        m = ceil(n3/minibatch);
        alpha = 1; rho = 100;
end
% Set learning rate
alpha_gd  = alpha;     alpha_sgd  = alpha/m;
alpha_pgd = rho*alpha; alpha_psgd = rho*alpha/m; 

rng(2,'twister'); 
X0 = randn(n,r);  

fgd = nan; fsgd = nan; fscgd = nan; fscsgd = nan;
ggd = nan; gsgd = nan; gscgd = nan; gscsgd = nan;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [~, fgd, ggd] = psd_gd(spmat, r, epochs, alpha_gd, momentum, X0, lossfun);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [~, fsgd, gsgd] = psd_sgd(spmat, r, epochs, alpha_sgd, lossfun, momentum, minibatch, [], X0);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ScaleGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [~, fscgd, gscgd] = psd_scalegd(spmat, r, epochs, alpha_pgd, momentum, X0, lossfun);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%
[~, fscsgd, gscsgd] = psd_scalesgd(spmat, r, epochs, alpha_psgd, lossfun, momentum, minibatch, [], X0);fprintf('\n')

%% Plot Figure
xlimit = inf;
figure;
subplot(211);
hold on
grid on
plot(0:numel(fgd)-1,fgd,'Color',Michigan_Yaize,'LineWidth',1.5); 
plot(0:numel(fsgd)-1,fsgd,'Color',Rackham_Green,'LineStyle','--','LineWidth',1.5); 
plot(0:numel(fscgd)-1,fscgd,'Color',Illini_Orange,'LineWidth',1.5);
plot(0:numel(fscsgd)-1,fscsgd,'Color',Illini_Blue,'LineStyle','--','LineWidth',1.5);
set(gca, 'yscale','log');
xlabel('Epochs','interpreter','latex');
xlim([0 xlimit])
switch lossfun
    case 'square'
        title('Square Loss')
        ylim([0.9*min([fgd,fsgd,fscgd,fscsgd]),inf])
        ylabel('$$f(X)$$','interpreter','latex');
        legend('GD','SGD','ScaleGD','ScaleSGD','location','sw');
    case '1bit'
        title('Pointwise Cross Entropy')
        ylim([0.9*min([fgd,fsgd,fscgd,fscsgd]),inf])
        ylabel('$$\|M-XX^T\|^2_F$$','interpreter','latex');
        legend('GD','SGD','ScaleGD','ScaleSGD','location','sw');
    case 'dist'
        title('Pairwise Square Loss for EDM')
        ylim([0.9*min([fgd,fsgd,fscgd,fscsgd]),inf])
        ylabel('$$f(X)$$','interpreter','latex');
        legend('GD','SGD','ScaleGD','ScaleSGD','location','sw');
    case 'pair'
        title('Pairwise Square Loss')
        ylim([0.9*min([fgd,fsgd,fscgd,fscsgd]),inf])
        ylabel('$$f(X)$$','interpreter','latex');
        legend('GD','SGD','ScaleGD','ScaleSGD','location','sw');
    case 'rank'
        title('Pairwise Hinge Rank Loss')
        ylim([0,1])
        ylabel('AUC','interpreter','latex');
        legend('GD','SGD','ScaleGD','ScaleSGD','location','sw');
    case 'ranklog'
        title('Pairwise Cross Entropy Rank Loss')
        ylim([0,1])
        ylabel('AUC','interpreter','latex');
        legend('GD','SGD','ScaleGD','ScaleSGD','location','sw');
end

subplot(212); 
hold on;
grid on
plot(0:numel(ggd)-1,ggd,'Color',Michigan_Yaize,'LineWidth',1.5); 
plot(0:numel(gsgd)-1,gsgd,'Color',Rackham_Green,'LineStyle','--','LineWidth',1.5); 
plot(0:numel(gscgd)-1,gscgd,'Color',Illini_Orange,'LineWidth',1.5);
plot(0:numel(gscsgd)-1,gscsgd,'Color',Illini_Blue,'LineStyle','--','LineWidth',1.5);
set(gca, 'yscale','log');
ylabel('$$\|\nabla f(X)\|$$','interpreter','latex');
legend('GD','SGD','ScaleGD','ScaleSGD','location','sw');
xlim([1 xlimit])
title('Gradient Norm')

%% Small Scale Testing Code
M = [1 0 2 0 4;
     0 0 1 2 4; 
     2 1 0 0 2;
     0 2 0 2 0;
     4 4 2 0 1];
r = 3;
epochs = 1000;
alpha =0.1;
momentum = 0;

% [U, fpsgd, gpsgd] = psd_sgd(M, r, epochs, alpha, 0, 1, [], 'ranklog');
[U, fpsgd, gpsgd] = psd_scalesgd(M, r, epochs, alpha, 'ranklog');
disp([M,U*U']);
