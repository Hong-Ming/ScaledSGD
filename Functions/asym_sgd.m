function [U,V,fhist,ghist] = asym_sgd(spmat, r, epochs, learning_rate, momentum, minibatch, U0, lossfun)
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
Threshold = 1e-15;
PrintFreq = 200;

% Robustness
[n,n1] = size(spmat);
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
    U = 0.000001*randn(n,r);
    V = 0.000001*randn(n1,r);
else
    U = reshape(X0(1:n*r),n,r); 
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
w2 = fprintf('* Solver: Asym_SGD,  Loss Function: %s loss',lossfun);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* search rank: %d, epochs: %d',r,epochs);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* learning_rate %3.1e, momentum: %3.1e, minibatch: %d',learning_rate,momentum,minibatch);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
w2 = fprintf('* numel(M): %d, nnz(M): %d, #sample: %d',numel(spmat),nnz(spmat),m);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');
[obj, gradU, gradV] = ComputeObjGrad(i,j,k,Y,m);
ini_fhist = obj;
ini_ghist = norm(gradU,'fro') + norm(gradV,'fro');
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
            [gradU, gradV] = ComputeGrad(bi,bj,bk,bY,batch_size);

            % Update latent factors
            if momentum == 0
                U = U - learning_rate*gradU;
                V = V - learning_rate*gradV;
            else
                Z = momentum*Z + grad;
                X = X - learning_rate*Z;
            end
        end
    else % no minibatch
        for idx = 1:m
            bi = i(idx); bj = j(idx); bk = k(idx); bY = Y(idx);
            % Compute gradient
            if strcmp(lossfun,'square')
                grad = U(bi,:)*V(bj,:)' - bk;
%             elseif strcmp(lossfun,'pair')
%                 grad = X(bi,:)*(X(bj,:)-X(bk,:))' - bY;
%             elseif strcmp(lossfun,'rank')
%                 zijk = bY*X(bi,:)*(X(bj,:)-X(bk,:))';
%                 grad = double(zijk < 0)*-bY;
%             elseif strcmp(lossfun,'ranklog')
%                 zijk = X(bi,:)*(X(bj,:)-X(bk,:))';
%                 grad = sigmoid(zijk)-bY;
            end

            % Update latent factors
            if strcmp(lossfun,'square')
                gradi = grad*V(bj,:);
                gradj = grad*U(bi,:);
                if momentum == 0
                    U(bi,:) = U(bi,:) - learning_rate*gradi;
                    V(bj,:) = V(bj,:) - learning_rate*gradj;
                else
                    Z(bi,:) = momentum*Z(bi,:) + gradi;
                    Z(bj,:) = momentum*Z(bj,:) + gradj;
                    U(bi,:) = U(bi,:) - learning_rate*Z(bi,:);
                    V(bj,:) = V(bj,:) - learning_rate*Z(bj,:);
                end
%             else
%                 gradi = grad*(X(bj,:)-X(bk,:));
%                 gradj = grad*X(bi,:);
%                 gradk = -gradj;
%                 if momentum == 0
%                     X(bi,:) = X(bi,:) - learning_rate*gradi;
%                     X(bj,:) = X(bj,:) - learning_rate*gradj;
%                     X(bk,:) = X(bk,:) - learning_rate*gradk;
%                 else
%                     Z(bi,:) = momentum*Z(bi,:) + gradi;
%                     Z(bj,:) = momentum*Z(bj,:) + gradj;
%                     Z(bk,:) = momentum*Z(bk,:) + gradk;
%                     X(bi,:) = X(bi,:) - learning_rate*Z(bi,:);
%                     X(bj,:) = X(bj,:) - learning_rate*Z(bj,:);
%                     X(bk,:) = X(bk,:) - learning_rate*Z(bk,:);
%                 end
            end
        end
    end

    % Print objective value and gradient norm
    [obj, gradU, gradV] = ComputeObjGrad(i,j,k,Y,m);
    fhist(epoch) = obj;
    ghist(epoch) = norm(gradU,'fro') + norm(gradV,'fro');
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
fhist = [ini_fhist, fhist(1:epoch)];
ghist = [ini_ghist, ghist(1:epoch)];

function [i,j,k,Y,m] = RetriveData()
    if strcmp(lossfun,'square')
        [i,j,k] = find(spmat);
        Y = zeros(1,numel(i));
    elseif strcmp(lossfun,'pair') || strcmp(lossfun,'rank') || strcmp(lossfun,'ranklog')
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
           if strcmp(lossfun,'pair')
               Y{row} = pairdif;
           elseif strcmp(lossfun,'rank')
               Y{row}(pairdif> 0) =  1;
               Y{row}(pairdif<=0) = -1;
           elseif strcmp(lossfun,'ranklog')
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
    if strcmp(lossfun,'square')
        fvec = zeros(m,1); E = zeros(m,1);
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx);
            RL = U(ii,:)*V(jj,:)' - kk;
            fvec(fdx) =  (1/2)*(RL)^2;
            E(fdx) = RL;
        end
        Eij = sparse(i,j,E,n,n1,m);
        obj = mean(fvec);
        gradU = (1/m)*(Eij*V);
        gradV = (1/m)*(Eij'*U);
%     elseif strcmp(lossfun,'pair')    
%         fvec = zeros(m,1); E = zeros(m,1);
%         for fdx = 1:m
%             ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
%             RL = X(ii,:)*(X(jj,:)-X(kk,:))'-Yijk;
%             fvec(fdx) = (1/2)*(RL)^2;
%             E(fdx) = RL; 
%         end
%         Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
%         obj =  mean(fvec);
%         grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
%     elseif strcmp(lossfun,'rank')
%         fvec = zeros(m,1); E = zeros(m,1);
%         for fdx = 1:m
%             ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
%             RL = Yijk*X(ii,:)*(X(jj,:)-X(kk,:))';
%             if RL < 0
%                 fvec(fdx) = -RL; E(fdx) = -Yijk;
%             end
%         end
%         Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
%         obj = mean(fvec);
%         grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
%     elseif strcmp(lossfun,'ranklog')
%         fvec = zeros(m,1); E = zeros(m,1);
%         for fdx = 1:m
%             ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
%             RL = X(ii,:)*(X(jj,:)-X(kk,:))';
%             if RL>0
%                 fvec(fdx) = log(1+exp(-RL))+RL-Yijk*RL;
%             else
%                 fvec(fdx) = log(1+exp(RL))-Yijk*RL;
%             end
%             E(fdx) = sigmoid(RL)-Yijk;
%         end
%         Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
%         obj = mean(fvec);
%         grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
    end
end

function [gradU, gradV] = ComputeGrad(i,j,k,Y,m) 
    if strcmp(lossfun,'square')
        E = zeros(m,1);
        for fdx = 1:m
            ii = i(fdx); jj = j(fdx); kk = k(fdx);
            E(fdx) = U(ii,:)*V(jj,:)' - kk;
        end
        Eij = sparse(i,j,E,n,n1,m);
        gradU = (1/m)*(Eij*V);
        gradV = (1/m)*(Eij'*U);
%     elseif strcmp(lossfun,'pair')
%         E = zeros(m,1);
%         for fdx = 1:m
%             ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
%             E(fdx) = X(ii,:)*(X(jj,:)-X(kk,:))'-Yijk;
%         end
%         Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
%         grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
%     elseif strcmp(lossfun,'rank')
%         E = zeros(m,1);
%         for fdx = 1:m
%             ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
%             RL = Yijk*X(ii,:)*(X(jj,:)-X(kk,:))';
%             if RL < 0
%                 E(fdx) = -Yijk;
%             end
%         end
%         Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
%         grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
%     elseif strcmp(lossfun,'ranklog')
%         E = zeros(m,1);
%         for fdx = 1:m
%             ii = i(fdx); jj = j(fdx); kk = k(fdx); Yijk = Y(fdx);
%             RL = X(ii,:)*(X(jj,:)-X(kk,:))';
%             E(fdx) = sigmoid(RL)-Yijk;
%         end
%         Eij = sparse(i,j,E,n,n,m); Eik = sparse(i,k,E,n,n,m);
%         grad = (1/m)*(Eij*X+Eij'*X-Eik*X-Eik'*X);
    end
end

end