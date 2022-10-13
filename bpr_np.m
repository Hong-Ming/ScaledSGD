function [best_auc] = bpr_np(spdata, d, epochs, learning_rate)
% Spdata must be supplied in four columns [i, j, k, Yijk] 
% with Yijk = 1 if item i is "closer" to item j than to item k, 
% and  Yijk = 0 if item i is "closer" to item k than to item j, 
% d is the total number of items
% r is the dimension of the underlying latent / feature space
% epochs is the total number of times to sweep the data.

% Parameter
Momentum = 0;
PrintFreq = 200;
WordCount = 0;

% Retrieve data
train_set = spdata.test(:,2:end);
test_set = spdata.test(:,2:end);

% Print info
x = randn(d,1); v = zeros(d,1);
w1 = fprintf(repmat('*',1,65));fprintf('*\n');
w2 = fprintf('* epochs: %d, learning rate: %3.1e',epochs,learning_rate);
fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
fprintf(repmat('*',1,65));fprintf('*\n');

[~, best_auc] = Evaluate(test_set, x);

for epoch = 1:epochs
    
    for idx = 1:size(train_set,1)        
        % Retrieve training data
        j = train_set(idx,1); k = train_set(idx,2); Yjk = train_set(idx,3);
        
        % Compute gradient (1 x Minibatch)
        zjk = x(j)-x(k); 
        grad = 1./(1+exp(-zjk))-Yjk;
                
        v(j) = Momentum*v(j) - grad;
        x(j) = x(j) + learning_rate*v(j);
        if j ~= k
            v(k) = Momentum*v(k) + grad;
            x(k) = x(k) + learning_rate*v(k);
        end
    end
    
    % Print objective value and gradient norm
    [ftest, auc] = Evaluate(test_set, x);
    if auc > best_auc; best_auc = auc; end
    fprintf(repmat('\b',1,WordCount));
    WordCount = fprintf('Epoch: %4d, Loss: %5.3e, AUC: %6.4f',epoch, ftest, auc);
    if mod(epoch,PrintFreq)==0; WordCount = fprintf('\n') - 1; end
end
if mod(epoch,PrintFreq)~=0; fprintf('\n'); end

end

function [obj, auc] = Evaluate(spdata,x)
j = spdata(:,1); k = spdata(:,2); 
Diff = x(j) - x(k);

% Evaluate function value
% objvec = log(1+exp(Diff)) - Yjk.*Diff;
Yjk = logical(spdata(:,3));
PosDiff = max(Diff,0);
objvec = PosDiff + log(exp(-PosDiff)+exp(Diff-PosDiff)) - Yjk.*Diff;
aucvec = ((Diff > 0) & Yjk) | ((Diff < 0) & ~Yjk);

% Output 
obj = mean(objvec); auc = mean(aucvec);
end
