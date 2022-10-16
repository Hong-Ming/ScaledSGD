function [X,fval] = scaledsgd(M, r, epochs, learning_rate, lossfun, DOSCALE)
% [X,Fval] = SCALEDSGD(M, r, epochs, learning_rate, lossfun, DOSCALE) 
% 
% SCALEDSGD   Scaled stochastic gradient descent algorithm for solving large, 
%             sparse, and symmetric matrix completion problem.
% 
% >  [X] = SCALEDSGD(M, r) performs stochastic gradient descent to compute an d x r 
%       factor X of M so that MAT = X*X' approximately satisfies MAT(i,j) = M(i,j) 
%       for all nonzero elements i,j in M. M must be square and should be large, 
%       sparse, and symmetric.
% 
% >  [X] = SCALEDSGD(M, r, epochs, learning_rate) specify the maximum number of
%       epochs (default to 1e3) and the learning rate (default to 1e-2).
% 
% >  [X] = SCALEDSGD(M, r, epochs, learning_rate, lossfun) specify the loss function.
%       Available loss function:
%           'RMSE' - (default) root mean square error.
%           '1bit' - pointwise cross-entropy loss.
%           'EDM'  - pairwise square loss for EDM completion.
% 
% >  [X] = SCALEDSGD(M, r, epochs, learning_rate, lossfun, DOSCALE) specify
%       wheater to apply scaling at each iteration (default to true). 
%       If DOSCALE = false, this algorithm is the same as stochastic
%       gradient descent (SGD).
% 
% >  [X, FVAL] = SCALEDSGD(...) also returns the history of residuals.

% Author: Hong-Ming Chiu (hmchiu2@illinois.edu)
% Date:   10 Oct 2022

% Polymorphism
if nargin < 3 || isempty(epochs);        epochs = 1e3;         end
if nargin < 4 || isempty(learning_rate); learning_rate = 1e-2; end
if nargin < 5 || isempty(lossfun);       lossfun = 'RMSE';     end
if nargin < 6 || isempty(DOSCALE);       DOSCALE = true;       end

% Input clean and check
[d,dchk] = size(M);
assert(d==dchk, '''M'' must be squrae.')
assert(mod(r,1) == 0 && r<=d && r > 0, 'Search rank ''r'' must be an integer and 0 < r <= d.')
assert(mod(epochs,1) == 0 && epochs > 0, '''epoch'' must be a positive integer.')
assert(learning_rate>0, '''learning_rate'' must be positive.')
assert(strcmp(lossfun,'RMSE') || strcmp(lossfun,'1bit') || strcmp(lossfun,'EDM'),...
       'Undefinied loss function ''lossfun''. Available loss function: ''RMSE'', ''1bit'', ''EDM''.')
assert(islogical(DOSCALE), '''DOSCALE'' must be logical.')

% Retrieve data
[i,j,val] = find(M);
spdata = [i(:),j(:),val(:)]; 
m = numel(i);

% Parameter
Threshold = 1e-16;             % error tolerance
X = randn(d,r);                % initial X
P = eye(r);                    % initail preconditioner
if DOSCALE; P = inv(X'*X); end % initail preconditioner
fval = inf(1,epochs);          % history of residuals
PrintFreq = 200;               % for display

% print info
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
if DOSCALE
    w2 = fprintf('* Solver: ScaledSGD,  Loss Function: %s loss',lossfun);
else
    w2 = fprintf('* Solver: SGD,  Loss Function: %s loss',lossfun);
end
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* search rank: %d, epochs: %d, learning rate: %3.1e',r,epochs,learning_rate);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* numel(M): %d, nnz(M): %d, #sample: %d',numel(M),nnz(M),m);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');

[ini_fval, grad] = ComputeObjGrad(spdata,X,lossfun);
WordCount = fprintf('Epoch: %4d, Loss: %8.4e, Grad: %8.4e',0, ini_fval, norm(grad,'fro'));

% Start SGD
for epoch = 1:epochs
    % shuffle data
    perm = randperm(m); spdata = spdata(perm,:);

    for idx = 1:m
        i = spdata(idx,1); j = spdata(idx,2);
        xi = X(i,:); xj = X(j,:); Y = spdata(idx,3);
        % Compute gradient
        switch lossfun
            case 'RMSE'
                grad = xi*xj' - Y;
                gradi = grad*xj*P;
                if i ~= j
                    gradj = grad*xi*P;
                end
            case '1bit'
                grad = sigmoid(xi*xj')-sigmoid(Y);
                gradi = grad*xj*P;
                if i ~= j
                    gradj = grad*xi*P;
                end
            case 'EDM'
                grad = xi*xi'+xj*xj'-2*xi*xj'-Y;
                gradi = grad*(xi-xj)*P;
                if i ~= j
                    gradj = -gradi;
                end
        end

        % Update latent factors       
        X(i,:) = xi - learning_rate*gradi;
        if i ~= j 
            X(j,:) = xj - learning_rate*gradj;
        end
        
        if DOSCALE
            % Update the pre-conditioner P
            Pu = P*X(i,:)'; p = X(i,:)*Pu; P = P - Pu*Pu' / (1+p);
            Pu = P*xi';     p = xi*Pu;     P = P + Pu*Pu' / (1-p);
            if i ~= j
                Pu = P*X(j,:)'; p = X(j,:)*Pu; P = P - Pu*Pu' / (1+p);
                Pu = P*xj';     p = xj*Pu;     P = P + Pu*Pu' / (1-p);
            end
        end
    end

    % Print objective value and gradient norm
    [fval(epoch), grad] = ComputeObjGrad(spdata,X,lossfun);
    fprintf(repmat('\b',1,WordCount));
    WordCount = fprintf('Epoch: %4d, Loss: %8.4e, Grad: %8.4e',epoch, fval(epoch), norm(grad,'fro'));
    
    if mod(epoch,PrintFreq)==0; WordCount = fprintf('\n') - 1; end
    if fval(epoch) <= Threshold; break; end
end
if mod(epoch,PrintFreq)~=0; fprintf('\n'); end
fprintf('\n');

% Output
fval(epoch+1:end) = fval(epoch);
fval = [ini_fval, fval];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [obj, grad] = ComputeObjGrad(spdata,X,lossfun)
i = spdata(:,1); j = spdata(:,2); Y = spdata(:,3);
d = size(X,1); m = numel(i); 
switch lossfun
    case 'RMSE'
        RL = sum(X(i,:).*X(j,:),2) - Y;
        Eij = sparse(i,j,RL,d,d,m);
        obj = mean((1/2)*RL.^2);
        grad = (1/m)*(Eij*X+Eij'*X);
    case '1bit'
        RL = sum(X(i,:).*X(j,:),2) - Y;
        Eij = sparse(i,j,sigmoid(RL)-sigmoid(Y),d,d,m);
        obj = mean((1/2)*RL.^2);
        grad = (1/m)*(Eij*X+Eij'*X);
    case 'EDM'
        RL = sum(X(i,:).^2,2) + sum(X(j,:).^2,2) - 2*sum(X(i,:).*X(j,:),2) - Y;
        Eij = sparse(i,j,RL,d,d,m); Eii = sparse(i,i,RL,d,d,m); Ejj = sparse(j,j,RL,d,d,m);
        obj = mean((1/4)*RL.^2);
        grad = (1/m)*(Eii*X+Ejj*X-Eij*X-Eij'*X);
end
end

function X_out = sigmoid(X_in)
X_out = 1./(1+exp(-X_in));
end





