%% color for plot
clc;clear
Starford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Michigan_Yaize = '#FFCB05';
Rackham_Green  = '#75988d';
Illini_Blue    = '#13294B';
x0=100;
y0=100;
width=500;
height=550;

%% Small Scale (500k pairwise comparisons) Fig 7
loader = load('TaskSmallDataFinal.mat');
fsgd = loader.ftrain1;
fscsgd = loader.ftrain2;
esgd = loader.etrain1;
escsgd = loader.etrain2;
esgdt = loader.etest1;
escsgdt = loader.etest2;
xlimit = inf;
figure;
hold on
grid on
plot(0:numel(fscsgd)-1,fscsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgd)-1,fsgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20)
title('Training BPR Loss','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
set(gcf,'position',[x0,y0,width,height])
% ylim([0.5 1])

figure;
hold on
grid on
plot(0:numel(escsgd)-1,escsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgd)-1,esgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20)
title('Training AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 1])
set(gcf,'position',[x0,y0,width,height])

figure;
hold on
grid on
plot(0:numel(escsgdt)-1,escsgdt,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgdt)-1,esgdt,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20)
title('Testing AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 1])
set(gcf,'position',[x0,y0,width,height])

%% Large Scale (2.6M pairwise comparisons) Fig 8
loader = load('TaskLargeDataFinal.mat');
fsgd = loader.ftrain1;
fscsgd = loader.ftrain2;
esgd = loader.etrain1;
escsgd = loader.etrain2;
esgdt = loader.etest1;
escsgdt = loader.etest2;
xlimit = inf;
figure;
hold on
grid on
plot(0:numel(fscsgd)-1,fscsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgd)-1,fsgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Training BPR Loss','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
% ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
set(gcf,'position',[x0,y0,width,height])

figure;
hold on
grid on
plot(0:numel(escsgd)-1,escsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgd)-1,esgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Training AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 0.65])
set(gcf,'position',[x0,y0,width,height])

figure;
hold on
grid on
plot(0:numel(escsgdt)-1,escsgdt,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgdt)-1,esgdt,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Testing AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 0.55])
set(gcf,'position',[x0,y0,width,height])
yticks([0.5,0.5125,0.525,0.5375,0.55])

%% Huge Scale (10M pairwise comparisons) Fig 9
loader = load('TaskLargeDataFinalPlots.mat');
fsgd = loader.ftrain1;
fscsgd = loader.ftrain2;
esgd = loader.etrain1;
escsgd = loader.etrain2;
esgdt = loader.etest1;
escsgdt = loader.etest2;
xlimit = inf;
figure;
hold on
grid on
plot(0:numel(fscsgd)-1,fscsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgd)-1,fsgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Training BPR Loss','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
% ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
set(gcf,'position',[x0,y0,width,height])
% ylim([0.5,1.5])

figure;
hold on
grid on
plot(0:numel(escsgd)-1,escsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgd)-1,esgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Training AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 0.75])
set(gcf,'position',[x0,y0,width,height])

figure;
hold on
grid on
plot(0:numel(escsgdt)-1,escsgdt,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgdt)-1,esgdt,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Testing AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 0.75])
set(gcf,'position',[x0,y0,width,height])

%% Small Scale with minibatch Fig 10
loader = load('TaskSmallFinalPlot64.mat');
fsgd = loader.ftrain1;
fscsgd = loader.ftrain2;
esgd = loader.etrain1;
escsgd = loader.etrain2;
esgdt = loader.etest1;
escsgdt = loader.etest2;
xlimit = inf;
figure;
hold on
grid on
plot(0:numel(fscsgd)-1,fscsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgd)-1,fsgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Training BPR Loss','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
% ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
set(gcf,'position',[x0,y0,width,height])

figure;
hold on
grid on
plot(0:numel(escsgd)-1,escsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgd)-1,esgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Training AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 0.75])
set(gcf,'position',[x0,y0,width,height])

figure;
hold on
grid on
plot(0:numel(escsgdt)-1,escsgdt,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgdt)-1,esgdt,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Testing AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 0.7])
yticks([0.5,0.55,0.6,0.65,0.7])
set(gcf,'position',[x0,y0,width,height])

%% Large Scale with minibatch Fig 11
loader = load('TaskLargeFinalPlot64.mat');
fsgd = loader.ftrain1;
fscsgd = loader.ftrain2;
esgd = loader.etrain1;
escsgd = loader.etrain2;
esgdt = loader.etest1;
escsgdt = loader.etest2;
xlimit = inf;
figure;
hold on
grid on
plot(0:numel(fscsgd)-1,fscsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgd)-1,fsgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Training BPR Loss','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
% ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
set(gcf,'position',[x0,y0,width,height])

figure;
hold on
grid on
plot(0:numel(escsgd)-1,escsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgd)-1,esgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Training AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 0.75])
set(gcf,'position',[x0,y0,width,height])

figure;
hold on
grid on
plot(0:numel(escsgdt)-1,escsgdt,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(esgdt)-1,esgdt,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Testing AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([0.5 0.75])
set(gcf,'position',[x0,y0,width,height])
