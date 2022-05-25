function plotfig1(fscsgd,fsgd,ascsgd,asgd,xlimit,percent)
% color for plot
Starford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Michigan_Yaize = '#FFCB05';
Rackham_Green  = '#75988d';
Illini_Blue    = '#13294B';

figure;
hold on
grid on
plot(0:numel(fscsgd)-1,fscsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgd)-1,fsgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title([num2str(percent) '$$\%$$ Samples'],'interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])

figure;
hold on
grid on
plot(0:numel(ascsgd)-1,ascsgd,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(asgd)-1,asgd,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title([num2str(percent) '$$\%$$ Samples'],'interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('AUC','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
end