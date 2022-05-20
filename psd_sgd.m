function [X,fhist,ghist] = psd_sgd(spmat, r, epochs, learning_rate, momentum, minibatch, X0, lossfun)
% PSD_SGD   Learn low-rank posdef matrix from samples of matrix elements
% X = PSD_SGD(SPMAT, R, K) performs K epochs of stochastic gradient descent
% to compute an N x R factor X so that MAT = X*X' approximately satisfies 
% MAT(i,j) = SPMAT(i,j) for all nonzero elements i,j in SPMAT. 
% SPMAT must be square and should be large, sparse, and symmetric.
% 
% X = PSD_SGD(SPMAT, R, K, LR) performs the above using learning rate LR.
% 
% X = PSD_SGD(SPMAT, R, K, LR, MO) performs the above using learning 
% rate LR and momentum MO.
% 
% X = PSD_SGD(SPMAT, R, K, LR, MO, MI) performs the above using learning 
% rate LR, momentum MO and minibatch size MI.
% 
% X = PSD_SGD(SPMAT, R, K, LR, MO, MI, X0) additionally uses X0 as the initial
% point, instead of the default random point.
% 
% X = PSD_SGD(SPMAT, R, K, LR, MO, MI, X0, lossfun) specify the loss function.
%   Available loss function:
%       'square'  - (default) square loss: sum_{i,j} |MAT(i,j) - SPMAT(i,j)|^2
%       'pair'    - pairwise square loss: sum_{i,j,k} |MAT(i,j)-MAT(i,k) - Y(i,j,k)|^2
%       'rank'    - pariwise hinge ranking loss
%       'ranklog' - pariwise corss entropy ranking loss
%       '1bit'    - pointwise 1 bit matrix completion loss
%       'dist'    - pairwise square loss for Euclidean distance matrix
% 
% [X, FHIST] = PSD_SGD(...) also returns the history of residuals
% 
% [X, FHIST, GHIST] = PSD_SGD(...) also returns the history of gradient
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
if nargin < 6 || isempty(minibatch)
    minibatch = 1;
end
if nargin < 7 || isempty(X0)
    X = randn(n,r);
else
    X = reshape(X0(1:n*r),n,r); 
end
if nargin < 8 || isempty(lossfun)
    lossfun = 'square';
end

% Retrieve data
[i,j,k,Y,m] = RetriveData;

% Check valid minibatch size
minibatch = floor(minibatch);
if minibatch>m
   warning('Minibatch size is greater than the number of sample, minibatch size will be set to be equal to number of sample') 
   minibatch = m;
elseif minibatch < 1
   warning('Minibatch size is lass than the 1, minibatch size will be set to be equal to 1') 
   minibatch = m;
end

fhist = inf(1,epochs); % history of residuals
ghist = inf(1,epochs); % history of gradient norm
Z = zeros(n,r);        % momentum storage
WordCount = 0;         % word counter

% print info
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
w2 = fprintf('* Solver: SGD,  Loss Function: %s loss',lossfun);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* search rank: %d, epochs: %d',r,epochs);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* learning_rate %3.1e, momentum: %3.1e, minibatch: %d',learning_rate,momentum,minibatch);
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
    % shuffle data
    perm = randperm(m);
    i = i(perm); j = j(perm); k = k(perm); Y = Y(perm);

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
            bk = k(batch_start:batch_end); bY = Y(batch_start:batch_end);
            grad = ComputeGrad(bi,bj,bk,bY,batch_size);

            % Update latent factors
            if momentum == 0
                X = X - learning_rate*grad;
            else
                Z = momentum*Z + grad;
                X = X - learning_rate*Z;
            end
        end
    else % no minibatch
        for idx = 1:m
            bi = i(idx); bj = j(idx); bk = k(idx); bY = Y(idx);
            % Compute gradient
            switch lossfun
                case 'square'
                    grad = X(bi,:)*X(bj,:)' - bY;
                case '1bit'
                    zijk = X(bi,:)*X(bj,:)';
                    grad = sigmoid(zijk)-bY;
                case 'dist'
                    grad = X(bi,:)*X(bi,:)'+X(bj,:)*X(bj,:)'-2*X(bi,:)*X(bj,:)'-bY;
                case 'pair'
                    grad = X(bi,:)*(X(bj,:)-X(bk,:))' - bY;
                case 'rank'
                    zijk = bY*X(bi,:)*(X(bj,:)-X(bk,:))';
                    grad = double(zijk < 0)*-bY;
                case 'ranklog'
                    zijk = X(bi,:)*(X(bj,:)-X(bk,:))';
                    grad = sigmoid(zijk)-bY;
            end

            % Update latent factors
            if strcmp(lossfun,'square') || strcmp(lossfun,'1bit') || strcmp(lossfun,'dist')  % Pointwise Loss
                if strcmp(lossfun,'dist')
                    gradi = grad*(X(bi,:)-X(bj,:));
                    gradj = grad*(X(bj,:)-X(bi,:));
                else
                    gradi = grad*X(bj,:);
                    gradj = grad*X(bi,:);
                end
                if momentum == 0
                    X(bi,:) = X(bi,:) - learning_rate*gradi;
                    X(bj,:) = X(bj,:) - learning_rate*gradj;
                else
                    Z(bi,:) = momentum*Z(bi,:) + gradi;
                    Z(bj,:) = momentum*Z(bj,:) + gradj;
                    X(bi,:) = X(bi,:) - learning_rate*Z(bi,:);
                    X(bj,:) = X(bj,:) - learning_rate*Z(bj,:);
                end
            elseif strcmp(lossfun,'pair') || strcmp(lossfun,'rank') || strcmp(lossfun,'ranklog') % Pairwise Loss
                gradi = grad*(X(bj,:)-X(bk,:));
                gradj = grad*X(bi,:);
                gradk = -gradj;
                if momentum == 0
                    X(bi,:) = X(bi,:) - learning_rate*gradi;
                    X(bj,:) = X(bj,:) - learning_rate*gradj;
                    X(bk,:) = X(bk,:) - learning_rate*gradk;
                else
                    Z(bi,:) = momentum*Z(bi,:) + gradi;
                    Z(bj,:) = momentum*Z(bj,:) + gradj;
                    Z(bk,:) = momentum*Z(bk,:) + gradk;
                    X(bi,:) = X(bi,:) - learning_rate*Z(bi,:);
                    X(bj,:) = X(bj,:) - learning_rate*Z(bj,:);
                    X(bk,:) = X(bk,:) - learning_rate*Z(bk,:);
                end
            end
        end
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

function grad = ComputeGrad(i,j,k,Y,m) 
E = zeros(m,1);
switch lossfun
    case 'square'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            E(fdx) = X(ii,:)*X(jj,:)' - Mij;
        end
        Eij = sparse(i,j,E,n,n,m);
        grad = (1/m)*(Eij*X+Eij'*X);
    case '1bit'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Yij = Y(fdx);
            RL = X(ii,:)*X(jj,:)';
            E(fdx) = sigmoid(RL)-Yij;
        end
        Eij = sparse(i,j,E,n,n,m);
        grad = (1/m)*(Eij*X+Eij'*X);
    case 'dist'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            RL = X(ii,:)*X(ii,:)'+X(jj,:)*X(jj,:)'-2*X(ii,:)*X(jj,:)' - Mij;
            E(fdx) = RL;
        end
        Eij = sparse(i,j,E,n,n,m); Eii = sparse(i,i,E,n,n,m); Ejj = sparse(j,j,E,n,n,m);
        grad = (1/m)*(Eii*X+Ejj*X-Eij*X-Eij'*X);
    case 'pair'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            E(fdx) = X(ii,:)*(X(jj,:)-X(kk,:))'-Yijk;
        end
        Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
        grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
    case 'rank'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = Yijk*X(ii,:)*(X(jj,:)-X(kk,:))';
            E(fdx) = double(RL<0)*-Yijk;
        end
        Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
        grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
    case 'ranklog'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = X(ii,:)*(X(jj,:)-X(kk,:))';
            E(fdx) = sigmoid(RL)-Yijk;
        end
        Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
        grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
end
end

function X_out = sigmoid(X_in)
X_out = 1./(1+exp(-X_in));
end

end