%% Load Data (Pairwise Cross Entropy Loss)
clc;clear
addpath Functions
% 30 x 30 symmetric well condtioned matrix, rank = 3
loader = load('Data/SYN_WELL30.mat');  
M_SynWell30 = loader.M;
% 30 x 30 symmetric ill condtioned matrix, rank = 3
loader = load('Data/SYN_ILL30.mat'); 
M_SynIll30 = loader.M;
% 100 x 100 symmetric item-item matrix from jester
loader = load('Data/JESTER.mat');
M_JESTER = loader.M;

% color for plot
Starford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Michigan_Yaize = '#FFCB05';
Rackham_Green  = '#75988d';
Illini_Blue    = '#13294B';

n = 30;
r = 3;

%% Show SGD works well for well conditon, ScaleSGD workds well for both well and ill condition (n^2 samples)

% Sample Ground Truth
% m = n^2;
n = 100;
M_SynIll30 = M_JESTER;
M_SynWell30 = M_JESTER;
m = 2*n*r;
spmat_SynWell30 = sampling(M_SynWell30,m);
spmat_SynIll30 = sampling(M_SynIll30,m);

epochs = 100;
learning_rate = 0.1;
lossfun = 'ranklog';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);  
[~, fsgdwell, ~] = psd_sgd(spmat_SynWell30, r, epochs, learning_rate, [], [], X0, lossfun);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);  
[~, fsgdill, ~] = psd_sgd(spmat_SynIll30, r, epochs, learning_rate, [], [], X0, lossfun);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);  
[~, fscsgdwell, ~] = psd_scalesgd(spmat_SynWell30, r, epochs, learning_rate, [], [], X0, lossfun);fprintf('\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); X0 = randn(n,r);  
[~, fscsgdill, ~] = psd_scalesgd(spmat_SynIll30, r, epochs, learning_rate, [], [], X0, lossfun);fprintf('\n')

% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = inf;

figure;
hold on
grid on
plot(0:numel(fscsgdwell)-1,fscsgdwell,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgdwell)-1,fsgdwell,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Well-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('AUC','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0,1])

figure;
hold on
grid on
plot(0:numel(fscsgdill)-1,fscsgdill,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgdill)-1,fsgdill,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Ill-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('AUC','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0,1])

%% Plot PrecSGD vs SGD (Well condition and Ill condition)
xlimit = inf;
figure;
hold on
grid on
plot(0:numel(fscsgdwell)-1,fscsgdwell,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgdwell)-1,fsgdwell,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Well-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('AUC','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0,1])

figure;
hold on
grid on
plot(0:numel(fscsgdill)-1,fscsgdill,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgdill)-1,fsgdill,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Ill-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('AUC','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0,1])