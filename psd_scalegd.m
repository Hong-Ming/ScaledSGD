function [X,fhist,ghist] = psd_scalegd(spmat, r, epochs, learning_rate, momentum, X0, lossfun)
% PSD_SCALEGD   Learn low-rank posdef matrix from samples of matrix elements
% X = PSD_SCALEGD(SPMAT, R, K) performs K steps of preconditioned gradient 
% descent to compute an N x R factor X so that MAT = X*X' approximately 
% satisfies MAT(i,j) = SPMAT(i,j) for all nonzero elements i,j in SPMAT. 
% SPMAT must be square and should be large, sparse, and symmetric.
% 
% X = PSD_SCALEGD(SPMAT, R, K, LR) performs the above using learning rate LR.
% 
% X = PSD_SCALEGD(SPMAT, R, K, LR, MO) performs the above using learning 
% rate LR and momentum MO.
% 
% X = PSD_SCALEGD(SPMAT, R, K, LR, MO, X0) additionally uses X0 as the initial
% point, instead of the default random point.
% 
% X = PSD_SCALEGD(SPMAT, R, K, LR, MO, X0, lossfun) specify the loss function.
%   Available loss function:
%       'square'  - (default) square loss: sum_{i,j} |MAT(i,j) - SPMAT(i,j)|^2
%       'pair'    - pairwise square loss: sum_{i,j,k} |MAT(i,j)-MAT(i,k) - Y(i,j,k)|^2
%       'rank'    - pariwise hinge ranking loss
%       'ranklog' - pariwise corss entropy ranking loss
%       '1bit'    - pointwise 1 bit matrix completion loss
%       'dist'    - pairwise square loss for Euclidean distance matrix
% 
% [X, FHIST] = PSD_SCALEGD(...) also returns the history of sum of square
% residuals sum_{i,j} |MAT(i,j) - SPMAT(i,j)|^2
% 
% [X, FHIST, GHIST] = PSD_SCALEGD(...) also returns the history of gradient
% norms 

% Parameter
Threshold = 1e-16;
PrintFreq = 200;

% Robustness
[n,nchk] = size(spmat);
if nchk ~= n, error('SPMAT must be squrae'); end
if nargin < 4 || isempty(learning_rate)
    learning_rate = 1e-2;
end
if nargin < 5 || isempty(momentum)
    momentum = 0;
end
if nargin < 6 || isempty(X0)
    X = randn(n,r);
else
    X = reshape(X0(1:n*r),n,r); 
end
if nargin < 7 || isempty(lossfun)
    lossfun = 'square';
end

% Retrieve data
[i,j,k,Y,m] = RetriveData;

fhist = inf(1,epochs); % history of residuals
ghist = inf(1,epochs); % history of gradient norm
Z = zeros(n,r);        % momentum storage
WordCount = 0;         % word counter

% print info
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
w2 = fprintf('* Solver: ScaleGD,  Loss Function: %s loss',lossfun);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* search rank: %d, epochs: %d',r,epochs);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* learning_rate %3.1e, momentum: %3.1e',learning_rate,momentum);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* numel(M): %d, nnz(M): %d, #sample: %d',numel(spmat),nnz(spmat),m);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');
[obj, grad] = ComputeObjGrad(i,j,k,Y,m);
ini_fhist = obj;
ini_ghist = norm(grad,'fro');
fprintf(repmat('\b',1,WordCount));
WordCount = fprintf('Epoch: %4d, Loss: %8.4e, Grad: %8.4e',0, ini_fhist, ini_ghist);

for epoch = 1:epochs
    % Shuffle data
    perm = randperm(m);
    i = i(perm); j = j(perm); k = k(perm); Y = Y(perm);
    
    % Compute gradient
%     grad = ComputePreGrad(i,j,k,Y,m);
    grad = grad/(X'*X);
    
    % Update latent factors
    if momentum == 0
        X = X - learning_rate*grad;
    else
        Z = momentum*Z + grad;
        X = X - learning_rate*Z;
    end
    
    % Print objective value and gradient norm
    [obj, grad] = ComputeObjGrad(i,j,k,Y,m);
    fhist(epoch) = obj;
    ghist(epoch) = norm(grad,'fro');
    fprintf(repmat('\b',1,WordCount));
    WordCount = fprintf('Epoch: %4d, Loss: %8.4e, Grad: %8.4e',epoch, fhist(epoch), ghist(epoch));
    if mod(epoch,PrintFreq)==0
        WordCount = 0;
        fprintf('\n')
    end
    if fhist(epoch) <= Threshold
        break
    end
end
if mod(epoch,PrintFreq)~=0
    fprintf('\n')
end

% Output
fhist(epoch+1:end) = fhist(epoch);
ghist(epoch+1:end) = ghist(epoch);
fhist = [ini_fhist, fhist];
ghist = [ini_ghist, ghist];

function [i,j,k,Y,m] = RetriveData()
if strcmp(lossfun,'square') || strcmp(lossfun,'1bit') || strcmp(lossfun,'dist') % Pointwise Loss
    [i,j,val] = find(spmat);
    k = ones(1,numel(i));
    switch lossfun
        case 'square'
            Y = val;
        case '1bit'
            Y = sigmoid(val);
        case 'dist'
            Y = val;        
    end                
elseif strcmp(lossfun,'pair') || strcmp(lossfun,'rank') || strcmp(lossfun,'ranklog') % Pairwise Loss
    i = cell(1,n); j = cell(1,n); k = cell(1,n); Y = cell(1,n);
    for row = 1:n
        thisrow = spmat(row,:);
        rr = find(thisrow);     % Find indices of nonzero elemt
        nr = numel(rr);
        [jdx,kdx] = find(tril(ones(nr))-1);
        i{row} = row*ones(1,nr*(nr-1)/2);
        j{row} = rr(jdx);
        k{row} = rr(kdx);
        pairdif = thisrow(j{row})-thisrow(k{row});
        switch lossfun
            case 'pair'
                Y{row} = pairdif;
            case 'rank'
                Y{row}(pairdif> 0) =  1;
                Y{row}(pairdif<=0) = -1;
            case 'ranklog'
                Y{row}(pairdif> 0) =  1;
                Y{row}(pairdif<=0) =  0;
        end
    end
    i = cell2mat(i); j = cell2mat(j); k = cell2mat(k); Y = cell2mat(Y);
else
     error('Choose a valid loss function');
end
m = numel(i);
end

function [obj, grad] = ComputeObjGrad(i,j,k,Y,m)
fvec = zeros(m,1); E = zeros(m,1);
switch lossfun
    case 'square'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            RL = X(ii,:)*X(jj,:)' - Mij;
            fvec(fdx) =  (1/2)*(RL)^2;
            E(fdx) = RL;
        end
        Eij = sparse(i,j,E,n,n,m);
        obj = mean(fvec);
        grad = (1/m)*(Eij*X+Eij'*X);
    case '1bit'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Yij = Y(fdx);
            RL = X(ii,:)*X(jj,:)';
            fvec(fdx) =  (1/2)*(RL-spmat(ii,jj))^2;
            E(fdx) = sigmoid(RL)-Yij;
        end
        Eij = sparse(i,j,E,n,n,m);
        obj = mean(fvec);
        grad = (1/m)*(Eij*X+Eij'*X);
    case 'dist'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            RL = X(ii,:)*X(ii,:)'+X(jj,:)*X(jj,:)'-2*X(ii,:)*X(jj,:)' - Mij;
            fvec(fdx) =  (1/4)*(RL)^2;
            E(fdx) = RL;
        end
        Eij = sparse(i,j,E,n,n,m); Eii = sparse(i,i,E,n,n,m); Ejj = sparse(j,j,E,n,n,m);
        obj = mean(fvec);
        grad = (1/m)*(Eii*X+Ejj*X-Eij*X-Eij'*X);
    case 'pair'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = X(ii,:)*(X(jj,:)-X(kk,:))'-Yijk;
            fvec(fdx) = (1/2)*(RL)^2;
            E(fdx) = RL; 
        end
        Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
        obj =  mean(fvec);
        grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
    case 'rank'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = Yijk*X(ii,:)*(X(jj,:)-X(kk,:))';
            fvec(fdx) = double(RL>=0);
            E(fdx) = double(RL<0)*-Yijk;
        end
        Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
        obj = mean(fvec);
        grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
    case 'ranklog'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = X(ii,:)*(X(jj,:)-X(kk,:))';
            fvec(fdx) = double((RL>0 && Yijk == 1) || (RL<=0 && Yijk == 0));
            E(fdx) = sigmoid(RL)-Yijk;
        end
        Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
        obj = mean(fvec);
        grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
end
end

function X_out = sigmoid(X_in)
X_out = 1./(1+exp(-X_in));
end

end