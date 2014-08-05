% Stochastic mirro-prox solver for the nonlinear game
%   min_x max_y .5<Px, x> + <Kx, y> - .5<Qy, y>
% where x and y are on simplices with same dimension, and we assume that
% max(max(abs(P))) == max(max(abs(Q))).

% References:
% [1] Nemirovski, A. (2004). Prox-method with rate of convergence O
% (1/t) for variational inequalities with Lipschitz continuous monotone
% operators and smooth convex-concave saddle point problems. SIAM Journal
% on Optimization, 15(1), 229-251.
%
% [2] Juditsky, A., Nemirovskii, A. S., & Tauvel, C. (2008). Solving
% variational inequalities with stochastic mirror-prox algorithm.
% Stochastic Systems, 1 (2011), 17-58.

function [xav, yav, etc] = funSMP(fhP, fhQ, fhK, fhKt, par)

n = par.n;
L = par.L;
M = par.M;
Omega = par.Omega;

MaxIter = funCheckPar(par, 'MaxIter', 300);
[bGapValue, fhGapValue] = funCheckPair(par, ...
    'bGapValue', 'fhGapValue');
GapEvaluationInterval = funCheckPar(par, 'GapEvaluationInterval', MaxIter);
nu = funCheckPar(par, 'nu', 1e-16);

% Constants for variational inequality
alpha = 1 + nu;

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.alpha = alpha;
etc.L = L;
etc.M = M;
etc.CPUTime = nan(MaxIter, 1);
etc.GapValue = nan(MaxIter, 1);
etc.PrimalObjectiveValue = nan(MaxIter, 1);
etc.DualObjectiveValue = nan(MaxIter, 1);
xnew = ones(n, 1) / n;
ynew = xnew;
xav = zeros(n, 1);
yav = xav;
Stepsize = min(alpha/(sqrt(3)*L), 2/M*sqrt(alpha*Omega^2/21/MaxIter));
etc.Stepsize = Stepsize;

tStart = tic;
for t = 1:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    % ------Variable updating
    y = ynew;
    x = xnew;
    
    % ------Extragradient step
    xeg = funProxMapEntropy(x, Stepsize * (fhP(x) + fhKt(y)));
    yeg = funProxMapEntropy(y, Stepsize * (fhQ(y) - fhK(x)));
    
    % ------Gradient step
    xnew = funProxMapEntropy(x, Stepsize * (fhP(xeg) + fhKt(yeg)));
    ynew = funProxMapEntropy(y, Stepsize * (fhQ(yeg) - fhK(xeg)));
    
    % ------Aggregate step
    xav = (xav*(t-1) + xeg)/t;
    yav = (yav*(t-1) + yeg)/t;
    
    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);
    
    % --------------------------------------
    % Calculate gap function
    % --------------------------------------
    if bGapValue
        if ~mod(t, GapEvaluationInterval)
            [etc.GapValue(t), etc.PrimalObjectiveValue(t), etc.DualObjectiveValue(t)]...
                = fhGapValue(xav, yav);
        end
    end
end

end