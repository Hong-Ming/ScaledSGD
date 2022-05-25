function [U,V,fhist,ghist,ahist] = asym_sgd(M, r, epochs, learning_rate, lossfun, momentum, minibatch, reg, U0, V0, M_true)
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
%       'pair'    - pairwise loss 
%       'rank'    - hinge ranking loss
%       'ranklog' - corss entropy ranking loss
% 
% [X, FHIST] = PSD_SGD(...) also returns the history of residuals
% 
% [X, FHIST, GHIST] = PSD_SGD(...) also returns the history of gradient
% norms 

% Parameter
Threshold = 1e-16;
PrintFreq = 200;

% Robustness
[n,n1] = size(M);
if nargin < 4 || isempty(learning_rate)
    learning_rate = 1e-2;
end
if nargin < 5 || isempty(lossfun)
    lossfun = 'square';
end
if nargin < 6 || isempty(momentum)
    momentum = 0;
end
if nargin < 7 || isempty(minibatch)
    minibatch = 1;
end
if nargin < 8 || isempty(reg)
    reg = 0;
end
if nargin < 9 || isempty(U0)
    U = randn(n,r);
else
    U = reshape(U0(1:n*r),n,r); 
end
if nargin < 10 || isempty(V0)
    V = randn(n1,r);
else
    V = reshape(V0(1:n1*r),n1,r); 
end
if nargin < 11 || isempty(M_true)
    M_true = M;
end

% Retrieve data
[i,j,k,Y,m] = RetriveData(M);
if nargout == 5
    [it,jt,kt,Yt,mt] = RetriveData(M_true);
end

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
ahist = inf(1,epochs); % history of auc score
ZU = zeros(n,r);        % momentum storage
ZV = zeros(n1,r);        % momentum storage
WordCount = 0;         % word counter

% print info
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
w2 = fprintf('* Solver: Asym_SGD,  Loss Function: %s loss',lossfun);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* search rank: %d, epochs: %d, learning rate: %3.1e',r,epochs,learning_rate);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* momentum: %3.1e, minibatch: %d, regularization: %3.1e',momentum,minibatch,reg);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* numel(M): %d, nnz(M): %d, #sample: %d',numel(M),nnz(M),m);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');
[obj, gradU, gradV] = ComputeObjGrad(i,j,k,Y,m);
ini_fhist = obj;
ini_ghist = norm(gradU,'fro') + norm(gradV,'fro');
if nargout == 5
    ini_ahist = Evaluation(it,jt,kt,Yt,mt);
    fprintf(repmat('\b',1,WordCount));
    WordCount = fprintf('Epoch: %4d, Loss: %8.4e, MSE: %6.4f, Grad: %8.4e',0, ini_fhist, ini_ahist, ini_ghist);
else
    ini_ahist = inf;
    fprintf(repmat('\b',1,WordCount));
    WordCount = fprintf('Epoch: %4d, Loss: %8.4e, Grad: %8.4e',0, ini_fhist, ini_ghist);
end

for epoch = 1:epochs
    % shuffle data
    perm = randperm(m);
    i = i(perm); j = j(perm); k = k(perm); Y = Y(perm);

    if minibatch > 1  || reg ~= 0  % with minibatch or regularization
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
            [gradU, gradV] = ComputeGrad(bi,bj,bk,bY,batch_size);

            % Update latent factors
            if momentum == 0
                U = U - learning_rate*gradU;
                V = V - learning_rate*gradV;
            else
                ZU = momentum*ZU + gradU;
                ZV = momentum*ZV + gradV;
                U = U - learning_rate*ZU;
                V = V - learning_rate*ZV;
            end
        end
    else % no minibatch
        for idx = 1:m
            bi = i(idx); bj = j(idx); bk = k(idx); bY = Y(idx);
            % Compute gradient
            switch lossfun
                case 'square'
                    grad = U(bi,:)*V(bj,:)' - bY;
                case '1bit'
                    zijk = U(bi,:)*V(bj,:)';
                    grad = sigmoid(zijk)-bY;
                case 'pair'
                    grad = U(bi,:)*(V(bj,:)-V(bk,:))' - bY;
                case 'rank'
                    zijk = bY*U(bi,:)*(V(bj,:)-V(bk,:))';
                    grad = double(zijk < 0)*-bY;
                case 'ranklog'
                    zijk = U(bi,:)*(V(bj,:)-V(bk,:))';
                    grad = sigmoid(zijk)-bY;
            end
            
            % Update latent factors
            if strcmp(lossfun,'square') || strcmp(lossfun,'1bit') % Pointwise Loss
                gradi = grad*V(bj,:);
                gradj = grad*U(bi,:);
                if momentum == 0
                    U(bi,:) = U(bi,:) - learning_rate*gradi;
                    V(bj,:) = V(bj,:) - learning_rate*gradj;
                else
                    ZU(bi,:) = momentum*ZU(bi,:) + gradi;
                    ZV(bj,:) = momentum*ZV(bj,:) + gradj;
                    U(bi,:) = U(bi,:) - learning_rate*ZU(bi,:);
                    V(bj,:) = V(bj,:) - learning_rate*ZV(bj,:);
                end
            elseif strcmp(lossfun,'pair') || strcmp(lossfun,'rank') || strcmp(lossfun,'ranklog') % Pairwise Loss
                gradi = grad*(V(bj,:)-V(bk,:));
                gradj = grad*U(bi,:);
                gradk = -gradj;
                if momentum == 0
                    U(bi,:) = U(bi,:) - learning_rate*gradi;
                    V(bj,:) = V(bj,:) - learning_rate*gradj;
                    V(bk,:) = V(bk,:) - learning_rate*gradk;
                else
                    ZU(bi,:) = momentum*ZU(bi,:) + gradi;
                    ZV(bj,:) = momentum*ZV(bj,:) + gradj;
                    ZV(bk,:) = momentum*ZV(bk,:) + gradk;
                    U(bi,:) = U(bi,:) - learning_rate*ZU(bi,:);
                    V(bj,:) = V(bj,:) - learning_rate*ZV(bj,:);
                    V(bk,:) = V(bk,:) - learning_rate*ZV(bk,:);
                end
            end
        end
    end

    % Print objective value and gradient norm
    [obj, gradU, gradV] = ComputeObjGrad(i,j,k,Y,m);
    fhist(epoch) = obj;
    ghist(epoch) = norm(gradU,'fro') + norm(gradV,'fro');
    if nargout == 5
        ahist(epoch) = Evaluation(it,jt,kt,Yt,mt);
        fprintf(repmat('\b',1,WordCount));
        WordCount = fprintf('Epoch: %4d, Loss: %8.4e, MSE: %6.4f, Grad: %8.4e',epoch, fhist(epoch), ahist(epoch), ghist(epoch));
    else
        fprintf(repmat('\b',1,WordCount));
        WordCount = fprintf('Epoch: %4d, Loss: %8.4e, Grad: %8.4e',epoch, fhist(epoch), ghist(epoch));
    end
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
ahist(epoch+1:end) = ahist(epoch);
fhist = [ini_fhist, fhist];
ghist = [ini_ghist, ghist];
ahist = [ini_ahist, ahist];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [i,j,k,Y,m] = RetriveData(M)
if strcmp(lossfun,'square') || strcmp(lossfun,'1bit') || strcmp(lossfun,'dist') % Pointwise Loss
    [i,j,val] = find(M);
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
        thisrow = M(row,:);
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

function [obj, gradU, gradV] = ComputeObjGrad(i,j,k,Y,m)
fvec = zeros(m,1); E = zeros(m,1);
switch lossfun
    case 'square'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            RL = U(ii,:)*V(jj,:)' - Mij;
            fvec(fdx) =  (1/2)*(RL)^2;
            E(fdx) = RL;
        end
        Eij = sparse(i,j,E,n,n1,m);
        obj = mean(fvec)+(reg/2)*(norm(U,'fro')^2+norm(V,'fro')^2);
        gradU = (1/m)*(Eij*V)+reg*U;
        gradV = (1/m)*(Eij'*U)+reg*V;
    case '1bit'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Yij = Y(fdx);
            RL = U(ii,:)*V(jj,:)';
            fvec(fdx) =  (1/2)*(RL-M(ii,jj))^2;
            E(fdx) = sigmoid(RL)-Yij;
        end
        Eij = sparse(i,j,E,n,n1,m);
        obj = mean(fvec)+(reg/2)*(norm(U,'fro')^2+norm(V,'fro')^2);
        gradU = (1/m)*(Eij*V)+reg*U;
        gradV = (1/m)*(Eij'*U)+reg*V;
    case 'pair'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = U(ii,:)*(V(jj,:)-V(kk,:))'-Yijk;
            fvec(fdx) = (1/2)*(RL)^2;
            E(fdx) = RL; 
        end
        Eij = sparse(i,j,E,n,n1,m); Eik = sparse(i,k,E,n,n1,m);
        obj =  mean(fvec)+(reg/2)*(norm(U,'fro')^2+norm(V,'fro')^2);
        gradU = (1/m)*(Eij*V-Eik*V)+reg*U;
        gradV = (1/m)*(Eij'*U-Eik'*U)+reg*V;
    case 'rank'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = Yijk*U(ii,:)*(V(jj,:)-V(kk,:))';
            fvec(fdx) = max(0,-RL);
            E(fdx) = double(RL<0)*-Yijk;
        end
        Eij = sparse(i,j,E,n,n1,m); Eik = sparse(i,k,E,n,n1,m);
        obj = mean(fvec)+(reg/2)*(norm(U,'fro')^2+norm(V,'fro')^2);
        gradU = (1/m)*(Eij*V-Eik*V)+reg*U;
        gradV = (1/m)*(Eij'*U-Eik'*U)+reg*V;
    case 'ranklog'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = U(ii,:)*(V(jj,:)-V(kk,:))';
            if RL > 0
                fvec(fdx) = log(1+exp(-RL))-(Yijk-1)*RL;
            else
                fvec(fdx) = log(1+exp(RL))-Yijk*RL;
            end
            E(fdx) = sigmoid(RL)-Yijk;
        end
        Eij = sparse(i,j,E,n,n1,m); Eik = sparse(i,k,E,n,n1,m);
        obj = mean(fvec)+(reg/2)*(norm(U,'fro')^2+norm(V,'fro')^2);
        gradU = (1/m)*(Eij*V-Eik*V)+reg*U;
        gradV = (1/m)*(Eij'*U-Eik'*U)+reg*V;
end
gradU = gradU/(U'*U);
gradV = gradV/(V'*V);
end

function [gradU, gradV] = ComputeGrad(i,j,k,Y,m) 
E = zeros(m,1);
switch lossfun
    case 'square'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            RL = U(ii,:)*V(jj,:)' - Mij;
            E(fdx) = RL;
        end
        Eij = sparse(i,j,E,n,n1,m);
        gradU = (1/m)*(Eij*V)+reg*U;
        gradV = (1/m)*(Eij'*U)+reg*V;
    case '1bit'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Yij = Y(fdx);
            RL = U(ii,:)*V(jj,:)';
            E(fdx) = sigmoid(RL)-Yij;
        end
        Eij = sparse(i,j,E,n,n1,m);
        gradU = (1/m)*(Eij*V)+reg*U;
        gradV = (1/m)*(Eij'*U)+reg*V;
    case 'pair'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = U(ii,:)*(V(jj,:)-V(kk,:))'-Yijk;
            E(fdx) = RL; 
        end
        Eij = sparse(i,j,E,n,n1,m); Eik = sparse(i,k,E,n,n1,m);
        gradU = (1/m)*(Eij*V-Eik*V)+reg*U;
        gradV = (1/m)*(Eij'*U-Eik'*U)+reg*V;
    case 'rank'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = Yijk*U(ii,:)*(V(jj,:)-V(kk,:))';
            E(fdx) = double(RL<0)*-Yijk;
        end
        Eij = sparse(i,j,E,n,n1,m); Eik = sparse(i,k,E,n,n1,m);
        gradU = (1/m)*(Eij*V-Eik*V)+reg*U;
        gradV = (1/m)*(Eij'*U-Eik'*U)+reg*V;
    case 'ranklog'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = U(ii,:)*(V(jj,:)-V(kk,:))';
            E(fdx) = sigmoid(RL)-Yijk;
        end
        Eij = sparse(i,j,E,n,n1,m); Eik = sparse(i,k,E,n,n1,m);
        gradU = (1/m)*(Eij*V-Eik*V)+reg*U;
        gradV = (1/m)*(Eij'*U-Eik'*U)+reg*V;
end
end

function eva = Evaluation(i,j,k,Y,m)
avec = zeros(1,m);
switch lossfun
    case 'square'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); Mij = Y(fdx);
            RL = U(ii,:)*V(jj,:)'-Mij;
            avec(fdx) = (RL)^2;
        end
        eva = sqrt(mean(avec));
    case 'rank'
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
            RL = U(ii,:)*(V(jj,:)-V(kk,:))';
            avec(fdx) = double((RL>0 && Yijk == 1) || (RL<=0 && Yijk == 0));
        end
        eva = mean(avec);
end
end

function X_out = sigmoid(X_in)
X_out = 1./(1+exp(-X_in));
end

end