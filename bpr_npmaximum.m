function [x, auc] = bpr_npmaximum(spdata, d, epochs, learning_rate)
% [X, AUC] = BPR_NPMAXIMUM(spdata, d, epochs, learning_rate) 
% 
% BPR_NPMAXIMUM   Compute a upper bound on the best AUC score by
%                 non-personalized ranking methods.
%                  
% >  [X] = BPR_NPMAXIMUM(spdata, d) performs stochastic gradient descent (SGD) 
%       to compute a upper bound on the best AUC score by minimizig the cross-entropy
%       loss between rankings in the item-item matrix M and the item vector x on
%       test set.
%       Inputs
%           - spdata.train: a data matrix of size m x 4. Each row of spdata.train is a 
%                pairwise measurements of the item-item matrix M, and its 4 entries are
%                given by [i,j,k,Yijk], where i, j and k are positive integer, and 
%                Yijk = 1 if item i is "closer" to item j than to item k, i.e., M(i,j) > M(i,k),
%                Yijk = 0 if item i is "closer" to item k than to item j, i.e., M(i,j) < M(i,k).
%           - spdata.test: a data matrix of size n x 4. The test set of spdata.train.
%           - d: the number of columns (or rows) in M.
%       Outputs
%           - x: a size d x 1 vector.
% 
% >  [X] = BPR_NPMAXIMUM(spdata, d, epochs, learning_rate) specify the maximum
%       number of epochs (default to 100) and the learning rate (default to 1e-1).
% 
% >  [X, AUC] = BPR_NPMAXIMUM(...) also returns the upper bound on the best AUC score.
% 

% Author: Hong-Ming Chiu (hmchiu2@illinois.edu)
% Date:   10 Oct 2022

% Polymorphism
if nargin < 3 || isempty(epochs);        epochs = 100;         end
if nargin < 4 || isempty(learning_rate); learning_rate = 1e-1; end

% Input clean and check
assert(mod(d,1) == 0 && d > 0, '''d'' must be a positive integer.')
assert(mod(epochs,1) == 0 && epochs > 0, '''epoch'' must be a positive integer.')
assert(learning_rate>0, '''learning_rate'' must be positive.')

% Retrieve data
train_set = spdata.test(:,2:end);
test_set = spdata.test(:,2:end);
m = size(train_set,1);

% Parameter
x = randn(d,1);    % initial x

% Print info
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
w2 = fprintf('* epochs: %d, learning rate: %3.1e',epochs,learning_rate);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');
[ftest, auc] = Evaluate(test_set, x);
WordCount = fprintf('Epoch: %4d, Loss: %5.3e, AUC: %6.4f',0, ftest, auc);

for epoch = 1:epochs
    
    train_set = train_set(randperm(m),:);
    for idx = 1:size(train_set,1)  
        
        % Retrieve training data
        j = train_set(idx,1); k = train_set(idx,2); Yjk = train_set(idx,3);
        
        % Compute gradient (1 x Minibatch)
        zjk = x(j)-x(k); 
        grad = 1./(1+exp(-zjk))-Yjk;
        
        % Update xj and xk
        x(j) = x(j) - learning_rate*grad;
        if j ~= k
            x(k) = x(k) + learning_rate*grad;
        end
    end
    
    % Print objective value and gradient norm
    [ftest, auc] = Evaluate(test_set, x);
    fprintf(repmat('\b',1,WordCount));
    WordCount = fprintf('Epoch: %4d, Loss: %5.3e, AUC: %6.4f',epoch, ftest, auc);
end
fprintf('\n\n')
end

function [obj, auc] = Evaluate(spdata, x)
j = spdata(:,1); k = spdata(:,2); 
Diff = x(j) - x(k);
Yjk = logical(spdata(:,3));

% Evaluate function value: objvec = log(1+exp(Diff)) - Yjk.*Diff;
PosDiff = max(Diff,0);
objvec = PosDiff + log(exp(-PosDiff)+exp(Diff-PosDiff)) - Yjk.*Diff;
obj = mean(objvec);

% Evaluate auc score
aucvec = ((Diff > 0) & Yjk) | ((Diff < 0) & ~Yjk);
auc = mean(aucvec);
end
