% AMP solver for the affine variational inequality (AVI) that computes u*
% such that
%   <F(u), u-u*> <= 0, for all u in Z,
% where 
%   F(u) = Au + b 
% is a monotone affine map.

% Note:
% - In this function, we assume that Z is a Lorentz cone.

function [wag, etc] = funA_amp(AG, AH, b, par)

n = size(AG, 2);
LipG = par.LipG;
LipH = par.LipH;

MaxIter = funCheckPar(par, 'MaxIter', 100);
bVerbose = funCheckPar(par, 'bVerbose', true);
[bGapValue, fhGapValue] = funCheckPair(par, ...
    'bGapValue', 'fhGapValue');

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.CPUTime = nan(MaxIter, 1);
etc.GapValue = nan(MaxIter, 1);
r = zeros(n, 1);
wag = zeros(n, 1);
% Stepsize
tlist = 1:MaxIter;
alpha = 2./(tlist + 1);
gamma = tlist./ (LipG + LipH.*MaxIter) ./ 3;

r1 = r;
tStart = tic;
for t = 1:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    % ------Middle step
    wmd = (1-alpha(t))*wag + alpha(t)*r;
    gradG = AG * wmd + b;
    
    % ------Extragradient step
    w = funProxMapLorentz(r, gamma(t)*(AH*r + gradG));
    
    % ------Gradient step
    r = funProxMapLorentz(r, gamma(t)*(AH*w + gradG));
    
    % ------Aggregate step
    wag = (1-alpha(t))*wag + alpha(t)*w;
    
    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);
end


% --------------------------------------
% Calculate perturbation
% --------------------------------------
etc.v = alpha(t)/gamma(t) * (r1 - r);
etc.vNorm = norm(etc.v);
funPrintf(bVerbose, 'Norm of perturbation: ||v||=%g', etc.vNorm);
funPrintf(bVerbose, '\n');
% --------------------------------------
% Calculate gap function
% --------------------------------------
if bGapValue
    etc.GapValue = fhGapValue(wag, etc.v);
    funPrintf(bVerbose, 'Gap value: %g', etc.GapValue);
    funPrintf(bVerbose, '\n');
end


end
