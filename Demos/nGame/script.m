%% Init
clear;
clc;
close all;
addpath('../../Utilities');

% Parameters
nPortfolio = 200; % Dimension of the portfolio of each player
MaxIter = 500;
seed = 18; % Random seed
bSave = false;
LipG = 10;
LipH = 1;
bVerbose = 0; % Silent/verbose mode

rng(seed, 'twister');
funPrintf(bVerbose, '%%-- %s --%% \r\n', datestr(now));
funPrintf(bVerbose, 'Seed = %g\r\n', seed);

for nPlayer = [5 20 50]
    fprintf('\n# of players:  %g\n', nPlayer);
    fprintf('Dimension of portfolio of each player: %g\n', nPortfolio);
    %% Generate n-person game
    fprintf('Generating problem...\n');
    n = nPortfolio * nPlayer;
    m = ceil(nPortfolio / 2);
    A = zeros(n, n);
    B = zeros(m * nPlayer, n);
    for i = 1 : nPlayer
        for j = i : nPlayer
            if i == j
                blkB = rand(m, nPortfolio);
                blkA = blkB' * blkB;
                scale = LipG / max(abs(blkA(:)));
                A((i-1)*nPortfolio + (1:nPortfolio), (j-1)*nPortfolio + (1:nPortfolio)) = blkA * scale;
                B((j-1)*m + (1:m), (i-1)*nPortfolio + (1:nPortfolio)) = blkB * sqrt(scale);
            else
                tmp = rand(nPortfolio);
                tmp = tmp - tmp.';
                tmp = tmp / max(abs(tmp(:))) * LipH;
                A((i-1)*nPortfolio + (1:nPortfolio), (j-1)*nPortfolio + (1:nPortfolio)) = tmp;
                A((j-1)*nPortfolio + (1:nPortfolio), (i-1)*nPortfolio + (1:nPortfolio)) = -tmp.';
            end
        end
    end
    
    AG = (A + A.') / 2;
    AH = (A - A.') / 2;
    
    fhGapValue = @(w)(funGn_GapValue(B, AH, w));
    
    %% Commonly used paramters
    par = [];
    par.bVerbose = bVerbose;
    par.MaxIter = MaxIter;
    par.bGapValue = 1;
    par.fhGapValue = fhGapValue;
    par.nPlayer = nPlayer;
    
    %% AMP
    funPrintf(bVerbose, 'Running AMP...\n');
    parAMP = par;
    [wAMP, etcAMP] = funGn_amp(AG, AH, parAMP);
    fprintf('CPU time: %g\n', etcAMP.CPUTime(end));
    
    %% MP
    funPrintf(bVerbose, 'Running MP...\n');
    parMP = par;
    parMP.MaxIter = par.MaxIter*2;
    [wMP, etcMP] = funGn_mp(A, parMP);
    fprintf('CPU time: %g\n', etcMP.CPUTime(end));
    
    %% Show results
    fprintf('\t\tAMP\t\tMP\n');
    fprintf('g0\t\t%e\t\t%e\n', etcAMP.GapValue, etcMP.GapValue);
    
    %% Save
    if bSave
        save(sprintf('nGame_%g_%g', nPlayer, nPortfolio));
    end
    
end