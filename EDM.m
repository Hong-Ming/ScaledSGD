%% Load Data (Euclidean Distance Matrix)
clc;clear
addpath Functions

loader = load('Data/EDM_WELL30.mat'); 
D_WELL = loader.D;
X_WELL = loader.X;

loader = load('Data/EDM_ILL30.mat'); 
D_ILL = loader.D;
X_ILL = loader.X;

% color for plot
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';

n = 30;
r = 3;

%% Show SGD works well for well conditon, PrecSGD workds well for both well and ill condition (n^2 samples)

% Sample Ground Truth
percent = 100;
spmat_WELL = sampling(D_WELL,percent);
spmat_ILL = sampling(D_ILL,percent);
epochs = 500;
learning_rate = 0.002;
lossfun = 'dist';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well SDG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fsgdwell] = sgd(spmat_WELL, r, epochs, 10*learning_rate, lossfun);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fsgdill] = sgd(spmat_ILL, r, epochs, learning_rate, lossfun);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Well ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fscsgdwell] = scaledsgd(spmat_WELL, r, epochs, 100*learning_rate, lossfun);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Syn Ill ScaleSGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); fprintf('\n')
[~, fscsgdill] = scaledsgd(spmat_ILL, r, epochs, 100*learning_rate, lossfun);

%% Plot ScaleSGD vs SGD (Well condition and Ill condition)
xlimit = inf;
figure;
hold on
grid on
plot(0:numel(fscsgdwell)-1,fscsgdwell,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgdwell)-1,fsgdwell,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Uniform in Cube (Well-Conditioned)','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
yticks([1e-18 1e-14 1e-10 1e-6 1e-2 1e2])
ylim([1e-18,1e2])

figure;
hold on
grid on
plot(0:numel(fscsgdill)-1,fscsgdill,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgdill)-1,fsgdill,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('With Outliers (Ill-Conditioned)','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
yticks([1e-18 1e-14 1e-10 1e-6 1e-2 1e2])
ylim([1e-18,1e2])

%% Plot Point Clouds
Starford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Michigan_Yaize = '#FFCB05';
Rackham_Green  = '#75988d';
Illini_Blue    = '#13294B';
coord = [ -1  -1  -1;
           1  -1  -1;
           1   1  -1;
          -1   1  -1;
          -1  -1   1;
           1  -1   1;
           1   1   1;
          -1   1   1 ];
idx = [4 8 5 1 4; 1 5 6 2 1; 2 6 7 3 2; 3 7 8 4 3; 5 8 7 6 5; 1 4 3 2 1]';
xc = coord(:,1);
yc = coord(:,2);
zc = coord(:,3);
figure
scatter3(X_WELL(:,1),X_WELL(:,2),X_WELL(:,3),20,'MarkerEdgeColor',Illini_Blue,'MarkerFaceColor',Illini_Blue)
grid on
hold on
box on
patch(xc(idx), yc(idx), zc(idx), 'r', 'facealpha', 0.03);
title('Sample Positions (Uniform in Cube)','interpreter','latex','FontSize',25);
xlabel('$$X$$','interpreter','latex','FontSize',25);
ylabel('$$Y$$','interpreter','latex','FontSize',25);
zlabel('$$Z$$','interpreter','latex','FontSize',25);
legend('Samples ','location','ne','FontSize',20);
xlim([-1,12])
ylim([-1,1])
zlim([-1,1])
view(3);

figure
scatter3(X_ILL(6:end,1),X_ILL(6:end,2),X_ILL(6:end,3),20,'MarkerEdgeColor',Illini_Blue,'MarkerFaceColor',Illini_Blue)
hold on
box on
grid on
scatter3(X_ILL(1:5,1),X_ILL(1:5,2),X_ILL(1:5,3),20,'MarkerEdgeColor',Illini_Orange,'MarkerFaceColor',Illini_Orange)
patch(xc(idx), yc(idx), zc(idx), 'r', 'facealpha', 0.03);
title('Sample Positions (With Outliers)','interpreter','latex','FontSize',25);
xlabel('$$X$$','interpreter','latex','FontSize',25);
ylabel('$$Y$$','interpreter','latex','FontSize',25);
zlabel('$$Z$$','interpreter','latex','FontSize',25);
legend('Samples','Outlier','location','ne','FontSize',20);
xlim([-1,12])
ylim([-1,1])
zlim([-1,1])
