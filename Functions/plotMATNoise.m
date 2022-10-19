function plotMATNoise(fscsgd_well,fsgd_well,fscsgd_ill,fsgd_ill,MW,MI,r,xlimit)
if nargin < 8; xlimit = inf; end

% Color for plot
Stanford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';

% compute noise floor
[U,S] = eig(MW); [~,idx] = sort(diag(S),'descend'); perm = idx(1:r);
U = U(:,perm); S = S(perm,perm);
nfw = (1/2)*(1/numel(MW))*norm(MW-U*S*U','fro')^2;

[U,S] = eig(MI); [~,idx] = sort(diag(S),'descend'); perm = idx(1:r);
U = U(:,perm); S = S(perm,perm);
nfi = (1/2)*(1/numel(MI))*norm(MI-U*S*U','fro')^2;

t = numel(fscsgd_well)-1;
nfw = nfw*ones(1,t+1);
nfi = nfi*ones(1,t+1);
ymax = 1.1*max([fscsgd_well(:);fsgd_well(:);fscsgd_ill(:);fsgd_ill(:)]);
ymin = 0.5*min([fscsgd_well(:);fsgd_well(:);fscsgd_ill(:);fsgd_ill(:);nfw(:);nfi(:)]);
figure;
hold on
grid on
plot(0:t,fscsgd_well,'Color',Illini_Orange,'LineWidth',2.5)
plot(0:t,fsgd_well,'Color',Illini_Blue,'LineWidth',2.5)
plot(0:t,nfw,'Color',Stanford_Red,'LineStyle','--','LineWidth',2)
set(gca,'fontsize',20,'yscale','log')
title('Well-conditioned','interpreter','latex','FontSize',25)
xlabel('Epochs','interpreter','latex','FontSize',25)
ylabel('$$f(X)$$','interpreter','latex','FontSize',25)
legend('ScaledSGD','SGD','Noise Floor','location','northeast','FontSize',25)
xlim([0 xlimit])
ylim([ymin,ymax])

figure;
hold on
grid on
plot(0:numel(fscsgd_ill)-1,fscsgd_ill,'Color',Illini_Orange,'LineWidth',2.5)
plot(0:numel(fsgd_ill)-1,fsgd_ill,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5)
plot(0:t,nfi,'Color',Stanford_Red,'LineStyle','--','LineWidth',2)
set(gca,'fontsize',20,'yscale','log')
title('Ill-conditioned','interpreter','latex','FontSize',25)
xlabel('Epochs','interpreter','latex','FontSize',25)
ylabel('$$f(X)$$','interpreter','latex','FontSize',25)
legend('ScaledSGD','SGD','Noise Floor','location','northeast','FontSize',25)
xlim([0 xlimit])
ylim([ymin,ymax])
end
