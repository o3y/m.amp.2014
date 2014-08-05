% Deterministic mirro-prox solver for multiple-player nonlinear game

% References: 
% [1] Nemirovski, A. (2004). Prox-method with rate of convergence O
% (1/t) for variational inequalities with Lipschitz continuous monotone
% operators and smooth convex-concave saddle point problems. SIAM Journal
% on Optimization, 15(1), 229-251.

function [zav, etc] = funGn_mp(A, par)

k = par.nPlayer;
n = size(A, 1)/k;

MaxIter = funCheckPar(par, 'MaxIter', 100);
bVerbose = funCheckPar(par, 'bVerbose', true);
[bGapValue, fhGapValue] = funCheckPair(par, ...
    'bGapValue', 'fhGapValue');
nu = funCheckPar(par, 'nu', 1e-16);
tolB = funCheckPar(par, 'BacktrackingTolerance', 1e-12);
L = funCheckPar(par, 'L0', nan);
alpha = 1 + nu;

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.alpha = alpha;
etc.CPUTime = nan(MaxIter, 1); 
etc.Stepsize = nan(MaxIter, 1);
etc.BacktrackingCount = nan(MaxIter, 1);
z = reshape(ones(k*n, 1) / n, [n, k]);
wav = zeros(n, k);

tStart = tic;
% --------------------------------------
% First iteration
% --------------------------------------
% Unless specified, generate L0 using the initial point and a random
% point
count = 0;
Fz = A*z(:);
while isnan(L) || isinf(L)
    count = count + 1;
    w = rand(n, k);
    w = bsxfun(@rdivide, w, sum(w));
    Fw = A*w(:);
    if isnan(L) || isinf(L)
        L = sqrt(sum(max((Fw-Fz).^2)) / sum(sum(abs(w - z).^2)));
    end
end
% Search for smallest L that satisfy the local Lipschitz condition for
% the first iteration
bStop = 0;
bFirstCorrectL = 0;
while 1
    count = count + 1;
    gamma = alpha/sqrt(2)/L;
    % ------Extragradient step
    w = funGn_ProxMapEntropy(z, gamma * Fz);
    Fw = A * w(:);
    % ------Backtracking for L
    if sum(max((Fw-Fz).^2)) > L * sum(sum(abs(w-z)).^2)
        L = L * 2;
        if bFirstCorrectL
            bStop = 1;
        end
        continue;
    else
        if ~bStop
            L = L / 2;
        end
    end
    bFirstCorrectL = 1;
    if bStop
        break;
    end
end
% ------Update
z = w;
zav = z;
% ------Save backtracking results
etc.gamma(1) = gamma;
etc.BacktrackingCount(1) = count;

gammaSum = gamma;
for t = 2:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------

    % ------Backtracking
    wnew = z;
    counter = 0;
    criterion = inf;
    while criterion>tolB
        if counter>3
            gamma = gamma / 2;
        end
        counter = counter + 1;
        w = wnew;
        Fw = A*w(:);
        wnew = funGn_ProxMapEntropy(z, gamma * Fw);
        criterion = gamma * ((w(:) - wnew(:)).' * Fw)   ...
            - sum(sum((wnew + nu/n).*log((wnew + nu/n)./(z + nu/n))));
    end
    z = wnew;
    zav = zav * gammaSum + gamma * z;
    gammaSum = gammaSum + gamma;
    zav = zav / gammaSum;
    etc.gamma(t) = gamma;
    etc.BacktrackingCount(t) = counter;
    if counter<=2
        gamma = gamma * 1.2;
    end
        
    funPrintf(bVerbose, 't=%g,\t', t);
    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);

    % --------------------------------------
    % Display backtracking results
    % --------------------------------------
    funPrintf(bVerbose, 'gamma=%g\tcounter=%g', ...
        etc.gamma(t), counter);
    
    funPrintf(bVerbose, '\n');
    
end
    
% --------------------------------------
% Calculate gap function
% --------------------------------------
if bGapValue
    etc.GapValue = fhGapValue(zav);
end

end