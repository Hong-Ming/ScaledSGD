function plotfig2(fsgd,fscsgd,aucsgd,aucscsgd,most_popular,np_max,xlimit)
% color for plot
Stanford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';
x0=100;
y0=100;
width=500;
height=550;

figure;
hold on
grid on
plot(0:numel(fscsgd.train)-1,fscsgd.train,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgd.train)-1,fsgd.train,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20)
title('Training BPR Loss','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
set(gcf,'position',[x0,y0,width,height])
ylim([0 inf])

figure;
hold on
grid on
plot(0:numel(aucscsgd.train)-1,aucscsgd.train,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(aucsgd.train)-1,aucsgd.train,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% plot(0:numel(fsgd)-1,np_max*ones(1,numel(fsgd)),'Color',Stanford_Red,'LineStyle','--','LineWidth',2);
% plot(0:numel(fsgd)-1,most_popular*ones(1,numel(fsgd)),'k','LineStyle','--','LineWidth',2);
set(gca,'fontsize',20)
title('Training AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([most_popular*0.95 1])
set(gcf,'position',[x0,y0,width,height])

figure;
hold on
grid on
plot(0:numel(aucscsgd.test)-1,aucscsgd.test,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(aucsgd.test)-1,aucsgd.test,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
plot(0:numel(fsgd.test)-1,np_max*ones(1,numel(fsgd.test)),'Color',Stanford_Red,'LineStyle','--','LineWidth',2);
plot(0:numel(fsgd.test)-1,most_popular*ones(1,numel(fsgd.test)),'k','LineStyle','--','LineWidth',2);
set(gca,'fontsize',20)
title('Testing AUC Score','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','np maximum','most popular','location','ne','FontSize',25);
xlim([0 xlimit])
ylim([most_popular*0.95 1])
set(gcf,'position',[x0,y0,width,height])
end