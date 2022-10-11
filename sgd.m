function [X,fval] = sgd(M, r, epochs, learning_rate, lossfun, momentum, minibatch)
% [X,Fval] = SGD(M, r, epochs, learning_rate, lossfun, momentum, minibatch) 
% 
% SGD   Stochastic gradient descent algorithm for solving large, sparse, 
%       and symmetric matrix completion problem.
% 
% >  [X] = SGD(M, r) performs stochastic gradient descent to compute an d x r 
%       factor X of M so that MAT = X*X' approximately satisfies MAT(i,j) = M(i,j) 
%       for all nonzero elements i,j in M. M must be square and should be large, 
%       sparse, and symmetric.
% 
% >  [X] = SGD(M, r, epochs, learning_rate) specify the maximum number of
%       epochs (default to 500) and the learning rate (default to 1e-2).
% 
% >  [X] = SGD(M, r, epochs, learning_rate, lossfun) specify the loss function.
%       Available loss function:
%           'RMSE' (default) - root mean square error.
%           'cross-entropy'  - pointwise cross-entropy loss.
%           'EDM'            - pairwise square loss for EDM completion.
% 
% >  [X] = SGD(M, r, epochs, learning_rate, lossfun, momentum, minibatch) also 
%       specify the momentum (default to 0) and minibatch size (default to 1).
% 
% >  [X, FVAL] = SGD(...) also returns the history of residuals.
% 

% Author: Hong-Ming Chiu (hmchiu2@illinois.edu)
% Date:   10 Oct 2022

% Polymorphism
if nargin < 3 || isempty(epochs);        epochs = 500;         end
if nargin < 4 || isempty(learning_rate); learning_rate = 1e-2; end
if nargin < 5 || isempty(lossfun);       lossfun = 'square';   end
if nargin < 6 || isempty(momentum);      momentum = 0;         end
if nargin < 7 || isempty(minibatch);     minibatch = 1;        end

% Input clean and check
[d,dchk] = size(M);
assert(d==dchk, '''M'' must be squrae.')
assert(mod(r,1) == 0 && r<=d && r > 0, 'Search rank ''r'' must be an integer and 0 < r <= d.')
assert(mod(epochs,1) == 0 && epochs > 0, '''epoch'' must be a positive integer.')
assert(learning_rate>0, '''learning_rate'' must be positive.')
assert(strcmp(lossfun,'square') || strcmp(lossfun,'1bit') || strcmp(lossfun,'dist'),...
       'Undefinied loss function ''lossfun''. Available loss function: ''square'', ''1bit'', ''dist''.')
assert(momentum>=0, '''momentum'' must be nonnegative.')
assert(mod(minibatch,1) == 0 && minibatch > 0, '''minibatch'' must be an positive integer.')

% Retrieve data
[i,j,Y,m] = RetriveData(M);

% Check valid minibatch size
minibatch = floor(minibatch);
if minibatch>m
   warning('Minibatch size is greater than the number of sample. Minibatch size is set to number of sample') 
   minibatch = m;
end

% Parameter
Threshold = 1e-16;     % error tolerance
X = randn(d,r);        % initial X
V = zeros(d,r);        % initial velocity of X
fval = inf(1,epochs);  % history of residuals
PrintFreq = 200;       % for display
WordCount = 0;         % word counter

% print info
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
w2 = fprintf('* Solver: SGD,  Loss Function: %s loss',lossfun);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* search rank: %d, epochs: %d, learning rate: %3.1e',r,epochs,learning_rate);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* momentum: %3.1e, minibatch: %d',momentum,minibatch);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* numel(M): %d, nnz(M): %d, #sample: %d',numel(M),nnz(M),m);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');

[ini_fval, grad] = ComputeObjGrad(i,j,Y,m);
fprintf(repmat('\b',1,WordCount));
WordCount = fprintf('Epoch: %4d, Loss: %8.4e, Grad: %8.4e',0, ini_fval, norm(grad,'fro'));

% Start SGD
for epoch = 1:epochs
    % shuffle data
    perm = randperm(m);
    i = i(perm); j = j(perm); Y = Y(perm);

    if minibatch > 1  % with minibatch
        for batch = 1:ceil(m/minibatch)  % loop over each batch
            batch_start = minibatch*(batch-1)+1;
            batch_end = minibatch*(batch);
            batch_size = minibatch;
            if batch == ceil(m/minibatch)  % take care of last batch
                batch_end = m;
                batch_size = batch_end-batch_start+1;
            end
            % compute gradient
            bi = i(batch_start:batch_end); bj = j(batch_start:batch_end); 
            bY = Y(batch_start:batch_end);
            grad = ComputeGrad(bi,bj,bY,batch_size);

            % Update latent factors
            V = momentum*V - grad;
            X = X + learning_rate*V;
        end
    else % no minibatch
        for idx = 1:m
            bi = i(idx); bj = j(idx); bY = Y(idx);
            % Compute gradient
            switch lossfun
                case 'square'
                    grad = X(bi,:)*X(bj,:)' - bY;
                    gradi = grad*X(bj,:);
                    if bi ~= bj
                        gradj = grad*X(bi,:);
                    end
                case '1bit'
                    zijk = X(bi,:)*X(bj,:)';
                    grad = sigmoid(zijk)-bY;
                    gradi = grad*X(bj,:);
                    if bi ~= bj
                        gradj = grad*X(bi,:);
                    end
                case 'dist'
                    grad = X(bi,:)*X(bi,:)'+X(bj,:)*X(bj,:)'-2*X(bi,:)*X(bj,:)'-bY;
                    gradi = 0;
                    if bi ~= bj
                        gradi = grad*(X(bi,:)-X(bj,:));
                        gradj = -gradi;
                    end
            end
            
            % Update latent factors           
            V(bi,:) = momentum*V(bi,:) - gradi;
            X(bi,:) = X(bi,:) + learning_rate*V(bi,:);
            if bi ~= bj
                V(bj,:) = momentum*V(bj,:) - gradj;
                X(bj,:) = X(bj,:) + learning_rate*V(bj,:);
            end
        end
    end

    % Print objective value and gradient norm
    [fval(epoch), grad] = ComputeObjGrad(i,j,Y,m);
    fprintf(repmat('\b',1,WordCount));
    WordCount = fprintf('Epoch: %4d, Loss: %8.4e, Grad: %8.4e',epoch, fval(epoch), norm(grad,'fro'));
    
    if mod(epoch,PrintFreq)==0; WordCount = fprintf('\n') - 1; end
    if fval(epoch) <= Threshold; break; end
end
if mod(epoch,PrintFreq)~=0; fprintf('\n'); end

% Output
fval(epoch+1:end) = fval(epoch);
fval = [ini_fval, fval];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [i,j,Y,m] = RetriveData(M)
[i,j,val] = find(M);
switch lossfun
    case 'square'
        Y = val;
    case '1bit'
        Y = sigmoid(val);
    case 'dist'
        Y = val;        
end                
m = numel(i);
end

function [obj, grad] = ComputeObjGrad(i,j,Y,m)
fvec = zeros(m,1); E = zeros(m,1);
switch lossfun
    case 'square'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            RL = X(ii,:)*X(jj,:)' - Mij;
            fvec(fdx) =  (1/2)*(RL)^2;
            E(fdx) = RL;
        end
        Eij = sparse(i,j,E,d,d,m);
        obj = mean(fvec);
        grad = (1/m)*(Eij*X+Eij'*X);
    case '1bit'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Yij = Y(fdx);
            RL = X(ii,:)*X(jj,:)';
            fvec(fdx) =  (1/2)*(RL-M(ii,jj))^2;
            E(fdx) = sigmoid(RL)-Yij;
        end
        Eij = sparse(i,j,E,d,d,m);
        obj = mean(fvec);
        grad = (1/m)*(Eij*X+Eij'*X);
    case 'dist'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            RL = X(ii,:)*X(ii,:)'+X(jj,:)*X(jj,:)'-2*X(ii,:)*X(jj,:)' - Mij;
            fvec(fdx) =  (1/4)*(RL)^2;
            E(fdx) = RL;
        end
        Eij = sparse(i,j,E,d,d,m); Eii = sparse(i,i,E,d,d,m); Ejj = sparse(j,j,E,d,d,m);
        obj = mean(fvec);
        grad = (1/m)*(Eii*X+Ejj*X-Eij*X-Eij'*X);
end
end

function grad = ComputeGrad(i,j,Y,m) 
E = zeros(m,1);
switch lossfun
    case 'square'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            E(fdx) = X(ii,:)*X(jj,:)' - Mij;
        end
        Eij = sparse(i,j,E,d,d,m);
        grad = (1/m)*(Eij*X+Eij'*X);
    case '1bit'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Yij = Y(fdx);
            RL = X(ii,:)*X(jj,:)';
            E(fdx) = sigmoid(RL)-Yij;
        end
        Eij = sparse(i,j,E,d,d,m);
        grad = (1/m)*(Eij*X+Eij'*X);
    case 'dist'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            RL = X(ii,:)*X(ii,:)'+X(jj,:)*X(jj,:)'-2*X(ii,:)*X(jj,:)' - Mij;
            E(fdx) = RL;
        end
        Eij = sparse(i,j,E,d,d,m); Eii = sparse(i,i,E,d,d,m); Ejj = sparse(j,j,E,d,d,m);
        grad = (1/m)*(Eii*X+Ejj*X-Eij*X-Eij'*X);
end
end

function X_out = sigmoid(X_in)
X_out = 1./(1+exp(-X_in));
end

end