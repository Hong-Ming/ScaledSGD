function [X,fval,auc] = bpr_scaledsgd(spdata, d, r, epochs, learning_rate, doScale)
% [X,Fval] = BRP_SCALEDSGD(spdata, d, r, epochs, learning_rate, momentum) 
% 
% BRP_SCALEDSGD   Scaled stochastic gradient descent algorithm for solving large, 
%                 sparse, and symmetric matrix completion problem.
% 
% >  [X] = BRP_SCALEDSGD(spdata, d, r) performs stochastic gradient descent to compute an d x r 
%       factor X of M.
% 
% >  [X] = BRP_SCALEDSGD(spdata, d, r, epochs, learning_rate) specify the maximum number of
%       epochs (default to 500) and the learning rate (default to 1e-2).
% 
% >  [X] = BRP_SCALEDSGD(M, r, epochs, learning_rate, lossfun, momentum, minibatch, reg) also 
%       specify the momentum (default to 0) and minibatch size (default to 1).
% 
% >  [X, FVAL] = BRP_SCALEDSGD(...) also returns the history of residuals.
% 
% >  [X, FVAL, AUC] = BRP_SCALEDSGD(...) also returns the history of auc score.
% 

% Author: Hong-Ming Chiu (hmchiu2@illinois.edu)
% Date:   10 Oct 2022

% Polymorphism
if nargin < 4 || isempty(epochs);        epochs = 50;          end
if nargin < 5 || isempty(learning_rate); learning_rate = 1e-2; end
if nargin < 6 || isempty(doScale);       doScale = true;       end

% Input clean and check
assert(mod(d,1) == 0 && d > 0, '''d'' must be a positive integer.')
assert(mod(r,1) == 0 && r<=d && r > 0, 'Search rank ''r'' must be an integer and 0 < r <= d.')
assert(mod(epochs,1) == 0 && epochs > 0, '''epoch'' must be a positive integer.')
assert(learning_rate>0, '''learning_rate'' must be positive.')
assert(islogical(doScale), '''doScale'' must be logical.')

% Retrieve data
train_set = spdata.train;
test_set = spdata.test;
m = size(train_set,1);

% Parameter
X = randn(d,r);                % initial X
P = eye(r);                    % initail preconditioner
if doScale; P = inv(X'*X); end % initail preconditioner
fval.train     = inf(1,epochs); % history of training loss at each epoch
fval.test      = inf(1,epochs); % history of training loss at each eopch
auc.train      = inf(1,epochs); % history of training auc score at each epoch
auc.test       = inf(1,epochs); % history of testing auc score at each epoch
fval.itertrain = inf(1,epochs*100); % history of training loss at each iteration
fval.itertest  = inf(1,epochs*100); % history of training loss at each iteration
auc.itertrain  = inf(1,epochs*100); % history of training auc score at each iteration
auc.itertest   = inf(1,epochs*100); % history of testing auc score at each iteration

% Print info
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
if doScale
    w2 = fprintf('* Solver: ScaledSGD,  Loss Function: BPR loss');
else
    w2 = fprintf('* Solver: SGD,  Loss Function: BPR loss');
end
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* search rank: %d, epochs: %d, learning rate: %3.1e',r,epochs,learning_rate);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');

[ini_ftrain, ini_auctrain] = Evaluate(train_set, X);
[ini_ftest, ini_auctest] = Evaluate(test_set, X);
fprintf('Epoch: %2d, Loss: %5.3e/%5.3e, AUC: %6.4f/%6.4f (train/test)\n',...
    0, ini_ftrain, ini_ftest, ini_auctrain, ini_auctest);
WordCount = 0;
iter = 1;
% Start ScaleSGD
for epoch = 1:epochs
    fprintf('Epoch: %2d, ', epoch)
    perm = randperm(m); train_set = train_set(perm,:);
    for idx = 1:m   
        % Retrieve training data
        i = train_set(idx,1); j = train_set(idx,2);
        k = train_set(idx,3); Yijk = train_set(idx,4);
        xi = X(i,:); xj = X(j,:); xk = X(k,:);
        % Compute gradient
        zijk = xi*xj'-xi*xk'; 
        grad = 1./(1+exp(-zijk))-Yijk;        
        gradjk = grad*xi*P;
        if i ~= j && i ~= k
            gradi = grad*(xj-xk)*P;
        end

        % Update latent factors
        X(j,:) = xj - learning_rate*gradjk;
        if k ~= j
            X(k,:) = xk + learning_rate*gradjk;
        end
        if i ~= j && i ~= k
            X(i,:) = xi - learning_rate*gradi;
        end
            
                
        if doScale            
            Pu = P*X(j,:)'; p = X(j,:)*Pu; P = P - Pu*Pu' / (1+p);
            Pu = P*xj';     p = xj*Pu;     P = P + Pu*Pu' / (1-p);
            if k ~= j
                Pu = P*X(k,:)'; p = X(k,:)*Pu; P = P - Pu*Pu' / (1+p);
                Pu = P*xk';     p = xk*Pu;     P = P + Pu*Pu' / (1-p);
            end
            if i ~= j && i ~= k
                Pu = P*X(i,:)'; p = X(i,:)*Pu; P = P - Pu*Pu' / (1+p);
                Pu = P*xi';     p = xi*Pu;     P = P + Pu*Pu' / (1-p);
            end
        end 
        if mod(idx, floor(m/100)) == 0
            [fval.itertrain(iter), ~] = Evaluate(train_set, X);
            [~,  auc.itertest(iter)]  = Evaluate(test_set, X, false);
            fprintf(repmat('\b',1,WordCount));
            WordCount = fprintf('Progress: %2d%% Loss: %5.3e, AUC: %6.4f',...
                floor(100*idx/m), fval.itertrain(iter), auc.itertest(iter));
            iter = iter + 1;
        end
    end
    
    % Print objective value and gradient norm
    [fval.train(epoch), auc.train(epoch)] = Evaluate(train_set, X);
    [fval.test(epoch),  auc.test(epoch)]  = Evaluate(test_set,  X);
    fprintf(repmat('\b',1,WordCount));
    fprintf('Loss: %5.3e/%5.3e, AUC: %6.4f/%6.4f (train/test)\n',...
        fval.train(epoch), fval.test(epoch), auc.train(epoch), auc.test(epoch));
    WordCount = 0;
end
fprintf('\n')

% Output
fval.train = [ini_ftrain, fval.train];
fval.test  = [ini_ftest, fval.test];
auc.train  = [ini_auctrain, auc.train];
auc.test   = [ini_auctest, auc.test];
fval.itertrain = [ini_ftrain, fval.itertrain];
fval.itertest  = [ini_ftest, fval.itertest];
auc.itertrain  = [ini_auctrain, auc.itertrain];
auc.itertest   = [ini_auctest, auc.itertest];
end

function [obj, auc] = Evaluate(spdata,X,evaobj)
obj = []; auc = [];
if nargin < 3; evaobj = true; end
% Efficiently compute M(i,j) - M(i,k) where M = X'*X
i = spdata(:,1); j = spdata(:,2); k = spdata(:,3); 
m = 1e6;
batchs = floor(numel(i)/m);
Diff = cell(batchs+1,1);
if batchs > 0
    ib = reshape(i(1:m*batchs),m,batchs); jb = reshape(j(1:m*batchs),m,batchs); 
    kb = reshape(k(1:m*batchs),m,batchs);
    i = i(m*batchs+1:end); j = j(m*batchs+1:end); k = k(m*batchs+1:end);
    for batch = 1:batchs
        Mij = sum(X(ib(:,batch),:).*X(jb(:,batch),:),2);
        Mik = sum(X(ib(:,batch),:).*X(kb(:,batch),:),2);
        Diff{batch} = Mij - Mik;
    end
end
Mij = sum(X(i,:).*X(j,:),2);
Mik = sum(X(i,:).*X(k,:),2);
Diff{batchs+1} = Mij - Mik;
Diff = cell2mat(Diff);
Yijk = logical(spdata(:,4));

if evaobj
    % Evaluate function value
    % objvec = log(1+exp(Diff)) - Yijk.*Diff;
    PosDiff = max(Diff,0);
    objvec = PosDiff + log(exp(-PosDiff)+exp(Diff-PosDiff)) - Yijk.*Diff;
    obj = mean(objvec);
end

if nargout > 1
    % Evaluate auc score value
    aucvec = ((Diff > 0) & Yijk) | ((Diff < 0) & ~Yijk);
    auc = mean(aucvec);
end

end