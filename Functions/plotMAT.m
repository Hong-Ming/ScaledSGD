function plotMAT(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill,xlimit)
if nargin < 5; xlimit = inf; end

% Colors for plot
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';

figure;
hold on
grid on
plot(0:numel(fscsgd_well)-1,fscsgd_well,'Color',Illini_Orange,'LineWidth',2.5)
plot(0:numel(fsgd_well)-1,fsgd_well,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5)
set(gca,'fontsize',20,'yscale','log')
title('Well-conditioned','interpreter','latex','FontSize',25)
xlabel('Epochs','interpreter','latex','FontSize',25)
ylabel('$$f(X)$$','interpreter','latex','FontSize',25)
legend('ScaledSGD','SGD','location','northeast','FontSize',25)
yticks([1e-18 1e-14 1e-10 1e-6 1e-2 1e2])
xlim([0 xlimit])
ylim([1e-18,1e2])

figure;
hold on
grid on
plot(0:numel(fscsgd_ill)-1,fscsgd_ill,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgd_ill)-1,fsgd_ill,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20,'yscale','log')
title('Ill-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','location','northeast','FontSize',25);
yticks([1e-18 1e-14 1e-10 1e-6 1e-2 1e2])
xlim([0 xlimit])
ylim([1e-18,1e2])
end
