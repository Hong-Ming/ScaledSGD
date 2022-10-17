function plotCFHuge(fscsgd,fsgd,aucscsgd,aucsgd,np_max)

% color for plot
Stanford_Red   = '#8C1515';
Illini_Orange  = '#DD3403';
Illini_Blue    = '#13294B';

t = numel(aucscsgd.itertest);
xlimit = 2*100;
epoch = numel(fsgd.train) - 1;
scale = 100;

figure;
hold on
grid on
ymax = max([fscsgd.itertrain,fsgd.itertrain]) + 0.1;
fill([0,scale,scale,0],[0,0,ymax,ymax],'w','FaceColor','#C84113','FaceAlpha',0.25,'EdgeColor','none')
fill(scale+[0,scale,scale,0],[0,0,ymax,ymax],'w','FaceColor','#1E3877','FaceAlpha',0.25,'EdgeColor','none')
h1 = plot(0:t-1,fscsgd.itertrain,'Color',Illini_Orange,'LineWidth',2.5);
h2 = plot(0:t-1,fsgd.itertrain,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20)
title('Training BPR Loss','interpreter','latex','FontSize',25)
xlabel('Epochs','interpreter','latex','FontSize',25)
ylabel('BPR Loss','interpreter','latex','FontSize',25)
legend([h1,h2],'ScaledSGD','SGD','location','ne','FontSize',25)
xlim([0 xlimit])
ylim([0.18 ymax])
xticks((0:epoch)*scale)
xticklabels({'0','1','2','3'})

figure;
hold on
grid on
tloc = 0.445;
pt = @(text) ['   ',num2str(text,'%.0f'),'%'];

fill([0,scale,scale,0],[0,0,1,1],'w','FaceColor','#C84113','FaceAlpha',0.25,'EdgeColor','none')
fill(scale+[0,scale,scale,0],[0,0,1,1],'w','FaceColor','#1E3877','FaceAlpha',0.25,'EdgeColor','none')

scsgd0 = InterX([0:t-1;aucscsgd.itertest],[0:t-1;np_max*ones(1,t)]);
plot(scsgd0(1)*ones(1,2),[0,scsgd0(2)],'Color',Illini_Orange,'LineStyle','--','LineWidth',1)
text(scsgd0(1),tloc,pt(scsgd0(1)),'Color',Illini_Orange,'FontSize',18,...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')

scsgd0 = InterX([0:t-1;aucscsgd.itertest],[0:t-1;0.9*ones(1,t)]);
plot(scsgd0(1)*ones(1,2),[0,scsgd0(2)],'Color',Illini_Orange,'LineStyle','--','LineWidth',1)
text(scsgd0(1),tloc,pt(scsgd0(1)-100),'Color',Illini_Orange,'FontSize',18,...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')

sgd0 = InterX([0:t-1;aucsgd.itertest],[0:t-1;np_max*ones(1,t)]);
plot(sgd0(1)*ones(1,2),[0,sgd0(2)],'Color',Illini_Blue,'LineStyle','--','LineWidth',1)
text(sgd0(1),tloc,pt(sgd0(1)),'Color',Illini_Blue,'FontSize',18,...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
        
sgd0 = InterX([0:t-1;aucsgd.itertest],[0:t-1;0.9*ones(1,t)]);
plot(sgd0(1)*ones(1,2),[0,sgd0(2)],'Color',Illini_Blue,'LineStyle','--','LineWidth',1)
text(sgd0(1),tloc,pt(sgd0(1)-100),'Color',Illini_Blue,'FontSize',18,...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')        
        
h3 = plot(0:t-1,np_max*ones(1,t),'Color',Stanford_Red,'LineStyle','--','LineWidth',2.5);
h1 = plot(0:t-1,aucscsgd.itertest,'Color',Illini_Orange,'LineWidth',2.5);
h2 = plot(0:t-1,aucsgd.itertest,'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
set(gca,'fontsize',20)
title('Testing AUC Score','interpreter','latex','FontSize',25)
xlabel('Epochs','interpreter','latex','FontSize',25)
ylabel('AUC Score','interpreter','latex','FontSize',25)
legend([h1,h2,h3],'ScaledSGD','SGD','NP-Maximum','location','se','FontSize',25)
xlim([0 xlimit])
ylim([0.49 1])
xticks((0:epoch)*scale)
xticklabels({'0','1','2','3'})

end

function P = InterX(L1,varargin)
%INTERX Intersection of curves
%   P = INTERX(L1,L2) returns the intersection points of two curves L1 
%   and L2. The curves L1,L2 can be either closed or open and are described
%   by two-row-matrices, where each row contains its x- and y- coordinates.
%   The intersection of groups of curves (e.g. contour lines, multiply 
%   connected regions etc) can also be computed by separating them with a
%   column of NaNs as for example
%
%         L  = [x11 x12 x13 ... NaN x21 x22 x23 ...;
%               y11 y12 y13 ... NaN y21 y22 y23 ...]
%
%   P has the same structure as L1 and L2, and its rows correspond to the
%   x- and y- coordinates of the intersection points of L1 and L2. If no
%   intersections are found, the returned P is empty.
%
%   P = INTERX(L1) returns the self-intersection points of L1. To keep
%   the code simple, the points at which the curve is tangent to itself are
%   not included. P = INTERX(L1,L1) returns all the points of the curve 
%   together with any self-intersection points.
%   
%   Example:
%       t = linspace(0,2*pi);
%       r1 = sin(4*t)+2;  x1 = r1.*cos(t); y1 = r1.*sin(t);
%       r2 = sin(8*t)+2;  x2 = r2.*cos(t); y2 = r2.*sin(t);
%       P = InterX([x1;y1],[x2;y2]);
%       plot(x1,y1,x2,y2,P(1,:),P(2,:),'ro')
%   Author : NS
%   Version: 3.0, 21 Sept. 2010
%   Two words about the algorithm: Most of the code is self-explanatory.
%   The only trick lies in the calculation of C1 and C2. To be brief, this
%   is essentially the two-dimensional analog of the condition that needs
%   to be satisfied by a function F(x) that has a zero in the interval
%   [a,b], namely
%           F(a)*F(b) <= 0
%   C1 and C2 exactly do this for each segment of curves 1 and 2
%   respectively. If this condition is satisfied simultaneously for two
%   segments then we know that they will cross at some point. 
%   Each factor of the 'C' arrays is essentially a matrix containing 
%   the numerators of the signed distances between points of one curve
%   and line segments of the other.
    %...Argument checks and assignment of L2
    error(nargchk(1,2,nargin));
    if nargin == 1,
        L2 = L1;    hF = @lt;   %...Avoid the inclusion of common points
    else
        L2 = varargin{1}; hF = @le;
    end
       
    %...Preliminary stuff
    x1  = L1(1,:)';  x2 = L2(1,:);
    y1  = L1(2,:)';  y2 = L2(2,:);
    dx1 = diff(x1); dy1 = diff(y1);
    dx2 = diff(x2); dy2 = diff(y2);
    
    %...Determine 'signed distances'   
    S1 = dx1.*y1(1:end-1) - dy1.*x1(1:end-1);
    S2 = dx2.*y2(1:end-1) - dy2.*x2(1:end-1);
    
    C1 = feval(hF,D(bsxfun(@times,dx1,y2)-bsxfun(@times,dy1,x2),S1),0);
    C2 = feval(hF,D((bsxfun(@times,y1,dx2)-bsxfun(@times,x1,dy2))',S2'),0)';
    %...Obtain the segments where an intersection is expected
    [i,j] = find(C1 & C2); 
    if isempty(i),P = zeros(2,0);return; end;
    
    %...Transpose and prepare for output
    i=i'; dx2=dx2'; dy2=dy2'; S2 = S2';
    L = dy2(j).*dx1(i) - dy1(i).*dx2(j);
    i = i(L~=0); j=j(L~=0); L=L(L~=0);  %...Avoid divisions by 0
    
    %...Solve system of eqs to get the common points
    P = unique([dx2(j).*S1(i) - dx1(i).*S2(j), ...
                dy2(j).*S1(i) - dy1(i).*S2(j)]./[L L],'rows')';
              
    function u = D(x,y)
        u = bsxfun(@minus,x(:,1:end-1),y).*bsxfun(@minus,x(:,2:end),y);
    end
end