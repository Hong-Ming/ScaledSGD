function plotCF(fscsgd,fsgd,aucscsgd,aucsgd,np_maximum,xlimit)
if nargin < 6; xlimit = inf; end
% color for plot
Stanford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';

t = numel(aucscsgd.itertest);
epoch = numel(fsgd.train) - 1;
scale = 100;
xlimit = xlimit*scale;
xtick = (0:epoch)*scale;
xticklabel = {'0'};
for i = 1:epoch
    xticklabel = [xticklabel,{num2str(i)}];
end

figure;
hold on
grid on
ymax = max([fscsgd.itertrain,fsgd.itertrain]) + 0.1;
ymin = 0.2;
h1 = plot(0:t-1,fscsgd.itertrain,'Color',Illini_Orange,'LineWidth',2.5);
h2 = plot(0:t-1,fsgd.itertrain,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20)
title('Training BPR Loss','interpreter','latex','FontSize',25)
xlabel('Epochs','interpreter','latex','FontSize',25)
ylabel('BPR Loss','interpreter','latex','FontSize',25)
legend([h1,h2],'ScaledSGD','SGD','location','ne','FontSize',25)
xlim([0 xlimit])
ylim([ymin ymax])
xticks(xtick)
xticklabels(xticklabel)

figure;
hold on
grid on
h3 = plot(0:t-1,np_maximum*ones(1,t),'Color',Stanford_Red,'LineStyle','--','LineWidth',2.5);
h1 = plot(0:t-1,aucscsgd.itertest,'Color',Illini_Orange,'LineWidth',2.5);
h2 = plot(0:t-1,aucsgd.itertest,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20)
title('Testing AUC Score','interpreter','latex','FontSize',25)
xlabel('Epochs','interpreter','latex','FontSize',25)
ylabel('AUC Score','interpreter','latex','FontSize',25)
legend([h1,h2,h3],'ScaledSGD','SGD','NP-Maximum','location','se','FontSize',25)
xlim([0 xlimit])
ylim([0.49 1])
xticks(xtick)
xticklabels(xticklabel)

end