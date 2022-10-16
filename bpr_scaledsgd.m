function [X, fval, auc] = bpr_scaledsgd(spdata, d, r, epochs, learning_rate, DOSCALE)
% [X, Fval, AUC] = BPR_SCALEDSGD(spdata, d, r, epochs, learning_rate, DOSCALE) 
% 
% BPR_SCALEDSGD   Scaled stochastic gradient descent (ScaledSGD) algorithm for
%                 training large-scale item-time collaborative filtering model.
% 
% >  [X] = BPR_SCALEDSGD(spdata, d, r) performs scaled stochastic gradient
%       descent (ScaledSGD) to compute an rank-r factorization X of an item-item
%       matrix M by minimizig the Bayesian Personalized Ranking (BPR) loss
%       between M and XX'. Let MAT = XX', the goal of BPR loss is the perserve the
%       ranking: if M(i,j) > M(i,k) then we want to have MAT(i,j) > MAT(i,k) for all (i,j,k)
%       Inputs
%           - spdata.train: a data matrix of size m x 4. Each row of spdata.train is a 
%                pairwise measurements of the item-item matrix M, and its 4 entries are
%                given by [i,j,k,Yijk], where i, j and k are positive integer, and 
%                Yijk = 1 if item i is "closer" to item j than to item k, i.e., M(i,j) > M(i,k),
%                Yijk = 0 if item i is "closer" to item k than to item j, i.e., M(i,j) < M(i,k).
%           - spdata.test: a data matrix of size n x 4. The test set of spdata.train.
%           - d: the number of items in M.
%           - r: search rank.
%       Outputs
%           - X: a size d x r factorization of M that preserve the ranking
%                between items.
% 
% >  [X] = BPR_SCALEDSGD(spdata, d, r, epochs, learning_rate) specify the maximum
%       number of epochs (default to 10) and the learning rate (default to 1e2).
% 
% >  [X] = BPR_SCALEDSGD(M, r, epochs, learning_rate, DOSCALE) specify
%       wheater to apply scaling at each stochastic update (default to true). 
%       If DOSCALE = false, this algorithm is the same as stochastic
%       gradient descent (SGD).
% 
% >  [X, FVAL] = BPR_SCALEDSGD(...) also returns the history of BPR loss.
% 
% >  [X, FVAL, AUC] = BPR_SCALEDSGD(...) also returns the history of AUC score.
% 

% Author: Hong-Ming Chiu (hmchiu2@illinois.edu)
% Date:   10 Oct 2022

% Polymorphism
if nargin < 4 || isempty(epochs);        epochs = 10;         end
if nargin < 5 || isempty(learning_rate); learning_rate = 1e2; end
if nargin < 6 || isempty(DOSCALE);       DOSCALE = true;      end

% Input clean and check
assert(mod(d,1) == 0 && d > 0, '''d'' must be a positive integer.')
assert(mod(r,1) == 0 && r<=d && r > 0, 'Search rank ''r'' must be an integer and 0 < r <= d.')
assert(mod(epochs,1) == 0 && epochs > 0, '''epoch'' must be a positive integer.')
assert(learning_rate>0, '''learning_rate'' must be positive.')
assert(islogical(DOSCALE), '''doScale'' must be logical.')

% Retrieve data
train_set = spdata.train;
test_set = spdata.test;
m = size(train_set,1);

% Parameter
X = randn(d,r);                     % initial X
P = eye(r);                         % initail preconditioner if DOSCALE = false
if DOSCALE; P = inv(X'*X); end      % initail preconditioner if DOSCALE = true
fval.train     = inf(1,epochs);     % history of training BPR loss at each epoch
fval.test      = inf(1,epochs);     % history of testing BPR loss at each eopch
fval.itertrain = inf(1,epochs*100); % history of training BPR loss at each iteration
auc.train      = inf(1,epochs);     % history of training AUC score at each epoch
auc.test       = inf(1,epochs);     % history of testing AUC score at each epoch
auc.itertest   = inf(1,epochs*100); % history of testing AUC score at each iteration

% Print info
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
if DOSCALE
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
iter = 1;

% Start ScaleSGD
for epoch = 1:epochs
    
    % Print info
    fprintf('Epoch: %2d, Progress: ', epoch)
    WordCount = fprintf('%2d%%',0);
    
    % Shuffle data
    train_set = train_set(randperm(m),:);
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
                       
        if DOSCALE
            % Update the pre-conditioner P by making six calls to
            % the Sherman-Morrison rank-1 update formula
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
            % Store traning BPR loss and testing AUC score at every 1% of epoch
            [fval.itertrain(iter), ~] = Evaluate(train_set, X, 'fvalonly');
            [~, auc.itertest(iter)]   = Evaluate(test_set, X, 'auclonly');
            fprintf(repmat('\b',1,WordCount));
            WordCount = fprintf('%2d%% Loss: %5.3e, AUC: %6.4f',...
                floor(100*idx/m), fval.itertrain(iter), auc.itertest(iter));
            iter = iter + 1;
        end
    end
    
    % Print BPR loss and AUC score
    [fval.train(epoch), auc.train(epoch)] = Evaluate(train_set, X);
    [fval.test(epoch),  auc.test(epoch)]  = Evaluate(test_set,  X);
    fprintf(repmat('\b',1,WordCount+10));
    fprintf('Loss: %5.3e/%5.3e, AUC: %6.4f/%6.4f (train/test)\n',...
        fval.train(epoch), fval.test(epoch), auc.train(epoch), auc.test(epoch));
end
fprintf('\n')

% Output
fval.train = [ini_ftrain, fval.train];
fval.test  = [ini_ftest, fval.test];
auc.train  = [ini_auctrain, auc.train];
auc.test   = [ini_auctest, auc.test];
fval.itertrain = [ini_ftrain, fval.itertrain];
auc.itertest   = [ini_auctest, auc.itertest];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [obj, auc] = Evaluate(spdata, X, options)
if nargin < 3; options = true; end
obj = []; auc = [];

% Efficiently compute M(i,j) - M(i,k) where M = X'*X. For large-scale
% problem, we split spdata into a batch of 1 million.
i = spdata(:,1); j = spdata(:,2); k = spdata(:,3); 
m = 1e6; batchs = floor(numel(i)/m);
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

if ~strcmp(options,'auconly')
    % Evaluate function value: objvec = log(1+exp(Diff)) - Yijk.*Diff;
    PosDiff = max(Diff,0);
    objvec = PosDiff + log(exp(-PosDiff)+exp(Diff-PosDiff)) - Yijk.*Diff;
    obj = mean(objvec);
end

if ~strcmp(options,'fvalonly')
    % Evaluate auc score
    aucvec = ((Diff > 0) & Yijk) | ((Diff < 0) & ~Yijk);
    auc = mean(aucvec);
end

end