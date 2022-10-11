function plotfig1(fscsgdwell,fsgdwell,fscsgdill,fsgdill,nfw,nfi,xlimit)
% Plot figure for RMSE and 1bit loss in noisy case
% color for plot
Stanford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';
n = numel(fscsgdwell)-1;
nfw = nfw*ones(1,n+1);
nfi = nfi*ones(1,n+1);
ymax = 1.1*max([fscsgdwell(:);fsgdwell(:);fscsgdill(:);fsgdill(:)]);
ymin = 0.5*min([fscsgdwell(:);fsgdwell(:);fscsgdill(:);fsgdill(:);nfw(:);nfi(:)]);
figure;
hold on
grid on
plot(0:n,fscsgdwell,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:n,fsgdwell,'Color',Illini_Blue,'LineWidth',2.5);
plot(0:n,nfw,'Color',Stanford_Red,'LineStyle','--','LineWidth',2);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Well-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','Noise Floor','location','ne','FontSize',25);
xlim([0 xlimit])
% yticks([1e-18 1e-14 1e-10 1e-6 1e-2 1e2])
ylim([ymin,ymax])

figure;
hold on
grid on
plot(0:numel(fscsgdill)-1,fscsgdill,'Color',Illini_Orange,'LineWidth',2.5);
plot(0:numel(fsgdill)-1,fsgdill,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
plot(0:n,nfi,'Color',Stanford_Red,'LineStyle','--','LineWidth',2);
set(gca, 'yscale','log');
set(gca,'fontsize',20)
title('Ill-conditioned','interpreter','latex','FontSize',25);
xlabel('Epochs','interpreter','latex','FontSize',25);
ylabel('$$f(X)$$','interpreter','latex','FontSize',25);
legend('ScaledSGD','SGD','Noise Floor','location','ne','FontSize',25);
xlim([0 xlimit])
% yticks([1e-18 1e-14 1e-10 1e-6 1e-2 1e2])
ylim([ymin,ymax])
end