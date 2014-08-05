% Stochastic accelerated mirro-prox solver for the nonlinear game
%   min_x max_y .5<Px, x> + <Kx, y> - .5<Qy, y>
% where x and y are on simplices with same dimension, and we assume that
% max(max(abs(P))) == max(max(abs(Q))).

% SMP in which the stepsize depends on N

function [xag, yag, etc] = funSAMP_N(fhP, fhQ, fhK, fhKt, par)

n = par.n;
LipG = par.LipG;
LipH = par.LipH;
Omega = par.Omega;
sigma = par.sigma;
beta = sigma / Omega;

MaxIter = funCheckPar(par, 'MaxIter', 100);
[bGapValue, fhGapValue] = funCheckPair(par, ...
    'bGapValue', 'fhGapValue');
GapEvaluationInterval = funCheckPar(par, 'GapEvaluationInterval', MaxIter);
nu = funCheckPar(par, 'nu', 1e-16);

% Constants for variational inequality
mu = 1 + nu;

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.mu = mu;
etc.CPUTime = nan(MaxIter, 1);
etc.gamma = nan(MaxIter, 1);
etc.GapValue = nan(MaxIter, 1);
etc.PrimalObjectiveValue = nan(MaxIter, 1);
etc.DualObjectiveValue = nan(MaxIter, 1);
xnew = ones(n, 1) / n;
ynew = xnew;
xag = zeros(n, 1);
yag = zeros(n, 1);

tStart = tic;
for t = 1:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    % ------Variable updating
    y = ynew;
    x = xnew;
    alpha = 2/(t+1);
    gamma = mu*t/(5*LipG + 3*LipH*MaxIter + beta*MaxIter*sqrt(mu*(MaxIter-1)));
    etc.gamma(t) = gamma;
    
    % ------Middle point
    xmd = (1 - alpha) * xag + alpha * x;
    ymd = (1 - alpha) * yag + alpha * y;
    Pxmd = fhP(xmd);
    Qymd = fhQ(ymd);
    
    % ------Extragradient step
    xeg = funProxMapEntropy(x, gamma * (Pxmd + fhKt(y)));
    yeg = funProxMapEntropy(y, gamma * (Qymd - fhK(x)));
    
    % ------Gradient step
    xnew = funProxMapEntropy(x, gamma * (Pxmd + fhKt(yeg)));
    ynew = funProxMapEntropy(y, gamma * (Qymd - fhK(xeg)));
    
    % ------Aggregate step
    xag = (1 - alpha) * xag + alpha * xnew;
    yag = (1 - alpha) * yag + alpha * ynew;
    
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
                = fhGapValue(xag, yag);
        end
    end
    
end
end