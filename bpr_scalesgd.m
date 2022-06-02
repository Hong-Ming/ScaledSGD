function [X,ftrain,ftest,etrain,etest,gradnrm] = bpr_scalesgd(spdata, n, r, epochs, learning_rate, doScale)
% Spdata must be supplied in four columns [i, j, k, Yijk] 
% with Yijk = 1 if item i is "closer" to item j than to item k, 
% and  Yijk = 0 if item i is "closer" to item k than to item j, 
% n is the total number of items
% r is the dimension of the underlying latent / feature space
% epochs is the total number of times to sweep the data.

% Parameter
Momentum = 0;
Minibatch = 64;
Threshold = 1e-16;
PrintFreq = 200;
test_size_size = round(0.02*size(spdata,1));

% Robustness
if nargin < 4 || isempty(learning_rate)
    learning_rate = 1e-2;
end
if nargin < 5 || isempty(doScale)
    doScale = true;
end

% Retrieve data
m = size(spdata,1);
perm = randperm(m);
test_set = spdata(perm(1:test_size_size),:);
train_set = spdata(perm(test_size_size+1:end),:);

ftrain = inf(1,epochs); % history of residuals
ftest = inf(1,epochs); % history of residuals
etrain = inf(1,epochs); % history of auc score
etest = inf(1,epochs); % history of residuals
gradnrm = inf(1,epochs); % history of gradient norm
WordCount = 0;         % word counter

% Print info
X = randn(n,r); P = inv(X'*X); V = zeros(n,r);
if ~doScale, P = eye(r); end
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
w2 = fprintf('* search rank: %d, epochs: %d, learning rate: %3.1e, scale: %d',r,epochs,learning_rate,doScale);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');

[ini_ftrain, ini_etrain, ini_gradnrm] = Evaluate(train_set, X);
[ini_ftest, ini_etest] = Evaluate(test_set, X);

% Start ScaleSGD
for epoch = 1:epochs
    % Shuffle data
    perm = randperm(m); spdata = spdata(perm,:);
    pointer = 1; m_train = size(train_set,1);
    
    while pointer <= m_train
        % Pointer logic for minibatch. Compute the end point and then
        pointer_end = min(pointer + Minibatch - 1, m_train);
        idx = pointer:pointer_end; pointer = pointer_end+1;
        
        % Retrieve training data
        i = train_set(idx,1); j = train_set(idx,2); k = train_set(idx,3); 
        Yijk = reshape(train_set(idx,4),1,[]); % must be row vec
        
        % Retrieve old latent factors
        xi = X(i,:)'; xj = X(j,:)'; xk = X(k,:)';
        if Momentum > 0
            vi = V(i,:)'; vj = V(j,:)'; vk = V(k,:)';
        end
        
        % Compute gradient (1 x Minibatch)
        zijk = sum(xi.*(xj-xk),1); 
        grad = 1./(1+exp(-zijk))-Yijk;

        % Update latent factors
        % note use of element-wise multiplication
        dui = P*(xj-xk); dujk = P*xi;
        if Momentum > 0
            vi_new = vi + grad.*dui;
            vj_new = vj + grad.*dujk;
            vk_new = vk + grad.*dujk;
            xi_new = xi - learning_rate*vi_new;
            xj_new = xj - learning_rate*vj_new;
            xk_new = xk - learning_rate*vk_new;
        else
            xi_new = xi - learning_rate*(grad.*dui);
            xj_new = xj - learning_rate*(grad.*dujk);
            xk_new = xk - learning_rate*(grad.*dujk);
        end
        
        if doScale
            if Minibatch > 1
                % With minibatch, the low-rank update is not low-rank, so
                % easier to just explicitly update
                P = inv(inv(P) - xi*xi' - xj*xj' - xk*xk' ...
                     + xi_new*xi_new' + xj_new*xj_new' + xk_new*xk_new');
            else
                % Update the Grammian inverse with new latent factors
                % inv(inv(P) + u*u') = P - P*u*u'*P / (1 + u'*P*u)
                % Note: this is explicitly unrolled to avoid overheads
                u = xi_new; Pu = P*xi_new; P = P - Pu*Pu' / (1+u'*Pu);
                u = xj_new; Pu = P*xj_new; P = P - Pu*Pu' / (1+u'*Pu);
                u = xk_new; Pu = P*xk_new; P = P - Pu*Pu' / (1+u'*Pu);

                % Downdate the Grammian inverse with old latent factors
                % inv(inv(P) - u*u') = P + P*u*u'*P / (1 - u'*P*u)
                u = xi; Pu = P*xi; P = P + Pu*Pu' / (1-u'*Pu);
                u = xj; Pu = P*xj; P = P + Pu*Pu' / (1-u'*Pu);
                u = xk; Pu = P*xk; P = P + Pu*Pu' / (1-u'*Pu);
            end
        end
        
        % Store new latent factors
        X(i,:) = xi_new'; X(j,:) = xj_new'; X(k,:) = xk_new';
        if Momentum > 0
            V(i,:) = vi_new'; V(j,:) = vj_new'; V(k,:) = vk_new';
        end
    end
    
    % Print objective value and gradient norm
    [ftrain(epoch), etrain(epoch), gradnrm(epoch)] = Evaluate(train_set, X);
    [ftest(epoch), etest(epoch)] = Evaluate(test_set, X);
    fprintf(repmat('\b',1,WordCount));
    WordCount = fprintf('Epoch: %4d, \n Loss(train): %8.4e, \n Loss(test): %8.4e, \n AUC(train): %6.4f, \n AUC(test): %6.4f \n Grad: %8.4e',epoch, ftrain(epoch), ftest(epoch), etrain(epoch), etest(epoch), gradnrm(epoch));
    if mod(epoch,PrintFreq)==0
        WordCount = 0;
        fprintf('\n')
    end
    if ftrain(epoch) <= Threshold
        break
    end
end
if mod(epoch,PrintFreq)~=0
    fprintf('\n')
end

% Output
ftrain(epoch+1:end) = ftrain(epoch);
ftest(epoch+1:end) = ftest(epoch);
etrain(epoch+1:end) = etrain(epoch);
etest(epoch+1:end) = etest(epoch);
gradnrm(epoch+1:end) = gradnrm(epoch);
ftrain = [ini_ftrain, ftrain];
ftest = [ini_ftest, ftest];
etrain = [ini_etrain, etrain];
etest = [ini_etest, etest];
gradnrm = [ini_gradnrm, gradnrm];
end

function [obj, err, gradnrm] = Evaluate(spdata,X)
% Efficiently compute M(i,j) - M(i,k) where M = X'*X
i = spdata(:,1); j = spdata(:,2); k = spdata(:,3); 
Mij = sum(X(i,:).*X(j,:),2); Mik = sum(X(i,:).*X(k,:),2);
Diff = Mij - Mik;

% Evaluate function value
% objvec = log(1+exp(Diff)) - Yijk.*Diff;
Yijk = logical(spdata(:,4));
PosDiff = max(Diff,0);
objvec = PosDiff + log(exp(-PosDiff)+exp(Diff-PosDiff)) - Yijk.*Diff;
gradvec = 1./(1+exp(-Diff)) - Yijk;
aucvec = ((Diff > 0) & Yijk) | ((Diff < 0) & ~Yijk);

% Output 
obj = mean(objvec); err = mean(aucvec); gradnrm = 0;

if nargout < 3, return; end
% Evaluate full gradient
grad = zeros(size(X)); 
m = size(spdata,1);
for idx = 1:m
    % Compute gradient update
    ii = i(idx); jj = j(idx); kk = k(idx); val = gradvec(idx);
    grad(ii,:) = grad(ii,:) + val*(X(jj,:)-X(kk,:));
    grad(jj,:) = grad(ii,:) + val*X(ii,:);
    grad(kk,:) = grad(ii,:) - val*X(ii,:);
end
gradnrm = norm(grad,'fro')/m; 
end