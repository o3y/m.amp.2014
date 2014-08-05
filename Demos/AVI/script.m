%% Init
clear;
% clc;
close all;
addpath('../../Utilities');

% Parameters
seed = 18; % Random seed
bVerbose = true; % Silent/verbose mode
bSave = false;

rng(seed, 'twister');
funPrintf(bVerbose, '%%-- %s --%% \r\n', datestr(now));
funPrintf(bVerbose, 'Seed = %g\r\n', seed);

for n = [1000, 5000, 10000]
    fprintf('n=%g\n', n);
    
    %% Generate AVI problem
    fprintf('Generating problem...\n');
    B = randn(ceil(n/2), n);
    AG = B'*B;
    tmp = rand(n);
    AH = tmp - tmp';
    A = AG+AH;
    b = rand(n, 1);
    
    LipG = eigs(AG, 1);
    LipH = sqrt(eigs(AH.'*AH, 1));
    LipA = sqrt(eigs(A.'*A, 1));
    
    fprintf('LipG=%g, LipH=%g\n', LipG, LipH);
    fprintf('\n');
    
    for MaxIter = [2000, 4000]
        fprintf('N=%g\n', MaxIter);
        %% Commonly used paramters
        par = [];
        par.bVerbose = bVerbose;
        par.MaxIter = MaxIter;
        par.bGapValue = false;
        
        %% AMP
        funPrintf(bVerbose, 'Running AMP...\n');
        parAMP = par;
        parAMP.LipG = LipG;
        parAMP.LipH = LipH;
        [wAMP, etcAMP] = funA_amp(AG, AH, b, parAMP);
        fprintf('CPU time: %g\n', etcAMP.CPUTime(end));
        
        %% MP
        funPrintf(bVerbose, 'Running MP...\n');
        parMP = par;
        parMP.LipA = LipA;
        parMP.MaxIter = par.MaxIter*2;
        [wMP, etcMP] = funA_mp(A, b, parMP);
        fprintf('CPU time: %g\n', etcMP.CPUTime(end));
        
        %% Calculate gap function values
        gvAMP = funGapLorentz(B, AH, b, wAMP, etcAMP.v);
        gvMP = funGapLorentz(B, AH, b, wMP, etcMP.v);
        
        fprintf('\t\tAMP\t\tMP\n');
        fprintf('|v|\t\t%e\t\t%e\n', etcAMP.vNorm, etcMP.vNorm);
        fprintf('gv\t\t%e\t\t%e\n', gvAMP, gvMP);
        fprintf('\n');
        
        %% Save
        if bSave
            save(sprintf('AVI_%g', n));
        end
    end
end