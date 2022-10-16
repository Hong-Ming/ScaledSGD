function plotMAT(fscsgdwell,fsgdwell,fscsgdill,fsgdill,xlimit)
if nargin < 5; xlimit = inf; end
% color for plot
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';
figure;
hold on
grid on
plot(0:numel(fscsgdwell)-1,fscsgdwell,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgdwell)-1,fsgdwell,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Well-conditioned','interpreter','latex','FontSize',25);
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
title('Ill-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','location','ne','FontSize',25);
xlim([0 xlimit])
yticks([1e-18 1e-14 1e-10 1e-6 1e-2 1e2])
ylim([1e-18,1e2])
end