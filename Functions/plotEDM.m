function plotEDM(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill,XW, XI, xlimit)
if nargin < 7; xlimit = inf; end

% Colors for plot
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';

figure;
hold on
grid on
plot(0:numel(fscsgd_well)-1,fscsgd_well,'Color',Illini_Orange,'LineWidth',2.5)
plot(0:numel(fsgd_well)-1,fsgd_well,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5)
set(gca,'fontsize',20,'yscale','log')
title('Uniform in Cube (Well-Conditioned)','interpreter','latex','FontSize',25)
xlabel('Epochs','interpreter','latex','FontSize',25)
ylabel('$$f(X)$$','interpreter','latex','FontSize',25)
legend('ScaledSGD','SGD','location','ne','FontSize',25)
yticks([1e-18 1e-14 1e-10 1e-6 1e-2 1e2])
xlim([0 xlimit])
ylim([1e-18,1e2])

figure;
hold on
grid on
plot(0:numel(fscsgd_ill)-1,fscsgd_ill,'Color',Illini_Orange,'LineWidth',2.5)
plot(0:numel(fsgd_ill)-1,fsgd_ill,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5)
set(gca,'fontsize',20,'yscale','log')
title('With Outliers (Ill-Conditioned)','interpreter','latex','FontSize',25)
xlabel('Epochs','interpreter','latex','FontSize',25)
ylabel('$$f(X)$$','interpreter','latex','FontSize',25)
legend('ScaledSGD','SGD','location','ne','FontSize',25)
yticks([1e-18 1e-14 1e-10 1e-6 1e-2 1e2])
xlim([0 xlimit])
ylim([1e-18,1e2])

% Plot Point Clouds
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

figure;
scatter3(XW(:,1),XW(:,2),XW(:,3),20,'MarkerEdgeColor',Illini_Blue,'MarkerFaceColor',Illini_Blue)
box on
grid on
hold on
patch(xc(idx), yc(idx), zc(idx), 'r', 'facealpha', 0.03)
title('Sample Positions (Uniform in Cube)','interpreter','latex','FontSize',25)
xlabel('$$X$$','interpreter','latex','FontSize',25)
ylabel('$$Y$$','interpreter','latex','FontSize',25)
zlabel('$$Z$$','interpreter','latex','FontSize',25)
legend('Samples ','location','ne','FontSize',20)
xlim([-1,12])
ylim([-1,1])
zlim([-1,1])
view(3);

figure;
scatter3(XI(6:end,1),XI(6:end,2),XI(6:end,3),20,'MarkerEdgeColor',Illini_Blue,'MarkerFaceColor',Illini_Blue)
box on
grid on
hold on
scatter3(XI(1:5,1),XI(1:5,2),XI(1:5,3),20,'MarkerEdgeColor',Illini_Orange,'MarkerFaceColor',Illini_Orange)
patch(xc(idx), yc(idx), zc(idx), 'r', 'facealpha', 0.03)
title('Sample Positions (With Outliers)','interpreter','latex','FontSize',25)
xlabel('$$X$$','interpreter','latex','FontSize',25)
ylabel('$$Y$$','interpreter','latex','FontSize',25)
zlabel('$$Z$$','interpreter','latex','FontSize',25)
legend('Samples','Outlier','location','ne','FontSize',20)
xlim([-1,12])
ylim([-1,1])
zlim([-1,1])
end
