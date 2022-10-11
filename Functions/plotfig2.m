function plotfig2(fsgd,fscsgd,esgd,escsgd,esgdt,escsgdt,xlimit)
% color for plot
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';
x0=100;
y0=100;
width=500;
height=550;

figure;
hold on
grid on
plot(0:numel(fscsgd)-1,fscsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgd)-1,fsgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20)
title('Training BPR Loss','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','location','ne','FontSize',25);
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
legend('ScaledSGD','SGD','location','se','FontSize',25);
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
legend('ScaledSGD','SGD','location','se','FontSize',25);
xlim([0 xlimit])
ylim([0.5 1])
set(gcf,'position',[x0,y0,width,height])
end