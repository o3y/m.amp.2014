% Stochastic mirro-prox solver for the nonlinear game
%   min_x max_y .5<Px, x> + <Kx, y> - .5<Qy, y>
% where x and y are on simplices with same dimension, and we assume that
% max(max(abs(P))) == max(max(abs(Q))).

% SAMP with alpha_t=1 for all t (equivalent to SMP)
function [xag, yag, etc] = funSAMP_NoAc(fhP, fhQ, fhK, fhKt, par)

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
    gamma = mu/(4*LipG + 3*LipH + beta*sqrt(mu*t));
    
    % Note: When LipG=0, the following choices of gamma are equivalent to that
    % of SMP
%     gamma = min(mu/(sqrt(3)*sqrt(2)*LipH), 2/(4*LipH)*sqrt(mu*Omega^2/21/MaxIter));
%     gamma = min(mu/(sqrt(3)*sqrt(2)*LipH), 1/(2*LipH)*sqrt(mu)*Omega/sqrt(MaxIter*21));
    % Min(_,_) type stepsize
%     gamma = min(mu/(3*LipH), 1/(2*LipH)*sqrt(mu)*Omega/sqrt(t));
    % Resemble SMP setting
%     gamma = 1/(3*LipH + (2*LipH)*sqrt(mu)*Omega/sqrt(t*21));
    
    etc.gamma(t) = gamma;
    
    % ------Middle point
    %     xmd = (1 - alpha) * xag + alpha * x;
    %     ymd = (1 - alpha) * yag + alpha * y;
    xmd = x;
    ymd = y;
    Pxmd = fhP(xmd);
    Qymd = fhQ(ymd);
    
    % ------Extragradient step
    xeg = funProxMapEntropy(x, gamma * (Pxmd + fhKt(y)));
    yeg = funProxMapEntropy(y, gamma * (Qymd - fhK(x)));
    
    % ------Gradient step
    xnew = funProxMapEntropy(x, gamma * (Pxmd + fhKt(yeg)));
    ynew = funProxMapEntropy(y, gamma * (Qymd - fhK(xeg)));
    
    % ------Aggregate step
    %     xag = (1 - alpha) * xag + alpha * xnew;
    %     yag = (1 - alpha) * yag + alpha * ynew;
    xag = (xag*(t-1) + xeg)/t;
    yag = (yag*(t-1) + yeg)/t;
    
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