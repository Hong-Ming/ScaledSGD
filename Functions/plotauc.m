function plotauc(ascsgdwell,asgdwell,ascsgdill,asgdill,xlimit)
% color for plot
Starford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Michigan_Yaize = '#FFCB05';
Rackham_Green  = '#75988d';
Illini_Blue    = '#13294B';
figure;
hold on
grid on
plot(0:numel(ascsgdwell)-1,ascsgdwell,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(asgdwell)-1,asgdwell,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Well-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('AUC','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
% ylim([0,1])

figure;
hold on
grid on
plot(0:numel(ascsgdill)-1,ascsgdill,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(asgdill)-1,asgdill,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
% set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Ill-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('AUC','interpreter','latex','FontSize',25);
legend('ScaleSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
% ylim([0,1])
end