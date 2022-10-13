function [X,fval,auc] = bpr_scaledsgd(spdata, d, r, epochs, learning_rate, doScale, momentum)
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
if nargin < 7 || isempty(momentum);      momentum = 0;         end

% Input clean and check
assert(mod(d,1) == 0 && d > 0, '''d'' must be a positive integer.')
assert(mod(r,1) == 0 && r<=d && r > 0, 'Search rank ''r'' must be an integer and 0 < r <= d.')
assert(mod(epochs,1) == 0 && epochs > 0, '''epoch'' must be a positive integer.')
assert(learning_rate>0, '''learning_rate'' must be positive.')
assert(islogical(doScale), '''doScale'' must be logical.')
assert(momentum>=0, '''momentum'' must be nonnegative.')

% Retrieve data
train_set = spdata.train;
test_set = spdata.test;
m = size(train_set,1);

% Parameter
X = randn(d,r);                % initial X
V = zeros(d,r);                % initial velocity of X
P = eye(r);                    % initail preconditioner
if doScale; P = inv(X'*X); end % initail preconditioner
fval.train = inf(1,epochs);    % history of residuals
fval.test  = inf(1,epochs);    % history of residuals
auc.train  = inf(1,epochs);    % history of auc score
auc.test   = inf(1,epochs);    % history of residuals

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
w2 = fprintf('* momentum: %3.1e, minibatch: %d',momentum,1);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');

[ini_ftrain, ini_auctrain] = Evaluate(train_set, X);
[ini_ftest, ini_auctest] = Evaluate(test_set, X);
WordCount = fprintf('Epoch: %d, Loss(train/test): %5.3e/%5.3e, AUC(train/test): %6.4f/%6.4f',...
    0, ini_ftrain, ini_ftest, ini_auctrain, ini_auctest);

% Start ScaleSGD
for epoch = 1:epochs
    perm = randperm(m); train_set = train_set(perm,:);
    for idx = 1:m   
        % Retrieve training data
        i = train_set(idx,1); j = train_set(idx,2);
        k = train_set(idx,3); Yijk = train_set(idx,4);
        
        % Compute gradient
        zijk = X(i,:)*(X(j,:)-X(k,:))'; 
        grad = 1./(1+exp(-zijk))-Yijk;
        
        gradjk = grad*X(i,:)*P;
        if i ~= j && i ~= k
            gradi = grad*(X(j,:)-X(k,:))*P;
        end

        % Update latent factors
        xj_old = X(j,:);
        V(j,:) = momentum*V(j,:) - gradjk;
        X(j,:) = xj_old + learning_rate*V(j,:);
        if k ~= j
            xk_old = X(k,:);
            V(k,:) = momentum*V(k,:) + gradjk;
            X(k,:) = xk_old + learning_rate*V(k,:);
        end
        if i ~= j && i ~= k
            xi_old = X(i,:);
            V(i,:) = momentum*V(i,:) - gradi;
            X(i,:) = xi_old + learning_rate*V(i,:);
        end
                
        if doScale            
            Pu = P*X(j,:)'; p = X(j,:)*Pu; P = P - Pu*Pu' / (1+p);
            Pu = P*xj_old'; p = xj_old*Pu; P = P + Pu*Pu' / (1-p);
            if k ~= j
                Pu = P*X(k,:)'; p = X(k,:)*Pu; P = P - Pu*Pu' / (1+p);
                Pu = P*xk_old'; p = xk_old*Pu; P = P + Pu*Pu' / (1-p);
            end
            if i ~= j && i ~= k
                Pu = P*X(i,:)'; p = X(i,:)*Pu; P = P - Pu*Pu' / (1+p);
                Pu = P*xi_old'; p = xi_old*Pu; P = P + Pu*Pu' / (1-p);
            end
        end 
    end
    
    % Print objective value and gradient norm
    [fval.train(epoch), auc.train(epoch)] = Evaluate(train_set, X);
    [fval.test(epoch),  auc.test(epoch)]  = Evaluate(test_set,  X);
    fprintf(repmat('\b',1,WordCount));
    WordCount = fprintf('Epoch: %d, Loss(train/test): %5.3e/%5.3e, AUC(train/test): %6.4f/%6.4f',...
        epoch, fval.train(epoch), fval.test(epoch), auc.train(epoch), auc.test(epoch));
end
fprintf('\n')

% Output
fval.train(epoch+1:end) = fval.train(epoch);
fval.test(epoch+1:end)  = fval.test(epoch);
auc.train(epoch+1:end)  = auc.train(epoch);
auc.test(epoch+1:end)   = auc.test(epoch);
fval.train = [ini_ftrain, fval.train];
fval.test  = [ini_ftest, fval.test];
auc.train  = [ini_auctrain, auc.train];
auc.test   = [ini_auctest, auc.test];
end

function [obj, err] = Evaluate(spdata,X)
% Efficiently compute M(i,j) - M(i,k) where M = X'*X
i = spdata(:,1); j = spdata(:,2); k = spdata(:,3); 
Mij = sum(X(i,:).*X(j,:),2); Mik = sum(X(i,:).*X(k,:),2);
Diff = Mij - Mik;

% Evaluate function value
% objvec = log(1+exp(Diff)) - Yijk.*Diff;
Yijk = logical(spdata(:,4));
PosDiff = max(Diff,0);
objvec = PosDiff + log(exp(-PosDiff)+exp(Diff-PosDiff)) - Yijk.*Diff;
aucvec = ((Diff > 0) & Yijk) | ((Diff < 0) & ~Yijk);

% Output 
obj = mean(objvec); err = mean(aucvec);
end