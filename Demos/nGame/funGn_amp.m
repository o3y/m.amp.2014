% Deterministic accelerated mirro-prox (AMP) solver for multiple-player 
% nonlinear game

function [wag, etc] = funGn_amp(AG, AH, par)

k = par.nPlayer;
n = size(AG, 1)/k;

MaxIter = funCheckPar(par, 'MaxIter', 100);
bVerbose = funCheckPar(par, 'bVerbose', true);
[bGapValue, fhGapValue] = funCheckPair(par, ...
    'bGapValue', 'fhGapValue');
nu = funCheckPar(par, 'nu', 1e-16);
mu = 1 + nu;
L = funCheckPar(par, 'L0', nan);
M = funCheckPar(par, 'M0', nan);

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.CPUTime = nan(MaxIter, 1);
etc.Lt = nan(MaxIter, 1);
etc.Mt = nan(MaxIter, 1);
etc.alpha = nan(MaxIter, 1);
etc.Gamma = nan(MaxIter, 1);
etc.BacktrackingCount = nan(MaxIter, 1);
r = reshape(ones(k*n, 1)/n, [n, k]);

tStart = tic;
% --------------------------------------
% First iteration
% --------------------------------------
% Unless specified, generate L0 and M0 using the initial point and a random
% point
Hr = AH*r(:);
gr = AG*r(:);
count = 0;
while isnan(L) || isinf(L) || isnan(M) || isinf(M)
    count = count + 1;
    wnew = rand(n, k);
    wnew = bsxfun(@rdivide, wnew, sum(wnew));
    gnew = AG*wnew(:);
    Hwnew = AH*wnew(:);
    if isnan(L) || isinf(L)
        L = (gnew - gr).'*(wnew(:) - r(:)) / sum(sum(abs(wnew - r).^2));
    end
    if isnan(M) || isinf(M)
        M = sum(max((Hwnew-Hr).^2)) / sum(sum(abs(wnew-r)).^2);
    end
end
% Search for smallest L and M that satisfy the backtracking condition for
% the first iteration
gmd = gr;
bStop = 0;
bFirstCorrectLM = 0;
while 1
    count = count + 1;
    gamma = mu/2/(L+M);
    % ------Extragradient step
    wnew = funGn_ProxMapEntropy(r, gamma*(gmd + Hr));
    gnew = AG*wnew(:);
    Hwnew = AH*wnew(:);
    % ------Backtracking for LipH
    if sum(max((Hwnew-Hr).^2)) > M * sum(sum(abs(wnew-r)).^2)
        M = M * 2;
        if bFirstCorrectLM
            bStop = 1;
        end
        continue;
    else
        if ~bStop
            M = M / 2;
        end
    end
    % ------Backtracking for LipG
    if (gnew - gr).'*(wnew(:) - r(:)) > L * sum(sum(abs(wnew - r)).^2)
        L = L * 2;
        if bFirstCorrectLM
            bStop = 1;
        end
        continue;
    else
        if ~bStop
            L = L / 2;
        end
    end
    bFirstCorrectLM = 1;
    if bStop
        break;
    end
end
% ------Update
r = funGn_ProxMapEntropy(r, gamma*(gmd + Hwnew));
Hr = AH*r(:);
gr = AG*r(:);
wag = wnew;
gag = gnew;
% ------Save backtracking results
etc.Lt(1) = L;
etc.Mt(1) = M;
etc.gamma(1) = gamma;
etc.BacktrackingCount(1) = count;

for t = 2:MaxIter
    alpha = 2/(t+1);
    
    % ------Gradient of G at the middle point wmd
    gmd = (1 - alpha) * gag + alpha * gr;

    % ------Backtracking
    count = 0;
    while 1
        count = count + 1;
        % ------Calculate stepsize
        gamma = mu*t/2/(L + M*t);
        
        % ------Extragradient step
        w = funGn_ProxMapEntropy(r, gamma*(gmd + Hr));
        g = AG*w(:);
        Hw = AH*w(:);
        
        % ------Backtracking for LipH
        if sum(max((Hw-Hr).^2)) > M^2 * sum(sum(abs(w-r)).^2)
            M = M * 2;
            continue;
        end
        
        % ------Backtracking for LipG
        if (g - gr).'*(w(:) - r(:)) > L * sum(sum(abs(w - r)).^2)
            L = L * 2;
            continue;
        end
        
        break;
    end
    
    % ------Update
    r = funGn_ProxMapEntropy(r, gamma*(gmd + Hw));
    Hr = AH*r(:);
    gr = AG*r(:);

    % ------Aggregate step
    wag = (1 - alpha) * wag + alpha * w;
    gag = (1 - alpha) * gag + alpha * g;
    
    % ------Save backtracking results
    etc.Lt(t) = L;
    etc.Mt(t) = M;
    etc.gamma(t) = gamma;
    etc.BacktrackingCount(t) = count;
    
    funPrintf(bVerbose, 't=%g,\t', t);
    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);
    
    % --------------------------------------
    % Display backtracking results
    % --------------------------------------
    funPrintf(bVerbose, 'L=%g\tM=%g\tcounter=%g\t', ...
        L, M, count);
    
    funPrintf(bVerbose, '\n');
    
end

% --------------------------------------
% Calculate gap function
% --------------------------------------
if bGapValue
    etc.GapValue = fhGapValue(wag);
end

end