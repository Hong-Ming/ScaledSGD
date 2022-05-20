function [U,V,fhist,ghist] = asym_gd(spmat, r, epochs, learning_rate, momentum, U0, V0)
% ASYM_GD   Learn low-rank posdef matrix from samples of matrix elements
% [U, V] = PSD_GD(SPMAT, R, K) performs K steps of gradient descent
% to compute an N x R factor U and M x R factor V so that MAT = U*V' approximately 
% satisfies MAT(i,j) = SPMAT(i,j) for all nonzero elements i,j in SPMAT. 
% 
% [U, V] = PSD_GD(SPMAT, R, K, LR) performs the above using learning rate LR.
% 
% [U, V] = PSD_GD(SPMAT, R, K, LR, MO) performs the above using learning 
% rate LR and momentum MO.

% [U, V] = PSD_GD(SPMAT, R, K, LR, MO, U0, V0) additionally uses U0 and V0 as 
% the initial point, instead of the default random point.

% [U, V, FHIST] = PSD_GD(...) also returns the history of sum of square
% residuals sum_{i,j} |MAT(i,j) - SPMAT(i,j)|^2

% [U, V, FHIST, GHIST] = PSD_GD(...) also returns the history of gradient
% norms 

% Parameter
Threshold = 1e-15;
PrintFreq = 200;

[n,n1] = size(spmat);
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
    U = randn(n,r);
    V = randn(n1,r);
else
    U = reshape(X0(1:n*r),n,r); 
end
if nargin < 8 || isempty(lossfun)
    lossfun = 'square';
end
fhist = inf(1,epochs);
ghist = inf(1,epochs);

% Retrieve data
[i,j,k,Y,m] = RetriveData;

% momentum storage
Z = zeros(n,r);
WordCount = 0;

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
    
    if momentum == 0
        U = U - learning_rate*gradU;
        V = V - learning_rate*gradV;
    else
        Z = momentum*Z + df;
        X = X - learning_rate*Z;
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
fprintf('\n')

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

end

