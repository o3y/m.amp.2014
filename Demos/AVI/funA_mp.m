% AMP solver for the affine variational inequality (AVI) that computes u*
% such that
%   <F(u), u-u*> <= 0, for all u in Z,
% where 
%   F(u) = Au + b 
% is a monotone affine map.

% References:
% [1] Nemirovski, A. (2004). Prox-method with rate of convergence O
% (1/t) for variational inequalities with Lipschitz continuous monotone
% operators and smooth convex-concave saddle point problems. SIAM Journal
% on Optimization, 15(1), 229-251.

% Note:
% - In this function, we assume that Z is a Lorentz cone.

function [wav, etc] = funA_mp(A, b, par)

n = size(A, 2);
LipA = par.LipA;

MaxIter = funCheckPar(par, 'MaxIter', 100);
bVerbose = funCheckPar(par, 'bVerbose', false);
[bGapValue, fhGapValue] = funCheckPair(par, ...
    'bGapValue', 'fhGapValue');

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.CPUTime = nan(MaxIter, 1);
etc.GapValue = nan(MaxIter, 1);
r = zeros(n, 1);
wav = zeros(n, 1);
% Stepsize
gamma = 1/LipA/sqrt(2);

r1 = r;
tStart = tic;
for t = 1:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    % ------Extragradient step
    w = funProxMapLorentz(r, gamma*(A*r + b));
    
    % ------Gradient step
    r = funProxMapLorentz(r, gamma*(A*w + b));
    
    % ------Aggregate step
    wav = (wav*(t-1) + w)/t;
    
    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);

end

% --------------------------------------
% Calculate perturbation
% --------------------------------------
etc.v = 1/t/gamma * (r1 - r);
etc.vNorm = norm(etc.v);
funPrintf(bVerbose, 'Norm of perturbation: ||v||=%g', etc.vNorm);
funPrintf(bVerbose, '\n');
% --------------------------------------
% Calculate gap function
% --------------------------------------
if bGapValue
    etc.GapValue = fhGapValue(wav, etc.v);
    funPrintf(bVerbose, 'Gap value: %g', etc.GapValue);
    funPrintf(bVerbose, '\n');
end


end
