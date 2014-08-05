%% Init
clear;
% clc;
close all;
addpath('../../Utilities');

% Parameters
seed = 18; % Random seed
bVerbose = true; % Silent/verbose mode
bSave = false; % Save screen outputs to a log file
sResultDir = '.';
nRun = 100;

if bSave
    if ~exist(sResultDir, 'dir')
        mkdir(sResultDir);
    end
    sLog = sprintf('%s/log_2game.txt', sResultDir);
    diary(sLog);
    diary on;
end
rng(seed, 'twister');
funPrintf(bVerbose, '%%-- %s --%% \r\n', datestr(now));
funPrintf(bVerbose, 'Seed = %g\r\n', seed);

for n = [1000, 2000, 5000]
    switch n
        case 1000
            L = 1;
        case 2000
            L = 10;
        case 5000
            L = 100;
    end
    
    %% Generate two person game
    fprintf('n=%g\n', n');
    fprintf('Generating data...\n');
    
    K = rand(n);
    dP = rand(ceil(n/2),n); P = dP' * dP; P = P .* (L/max(abs(P(:))));
    dQ = rand(ceil(n/2),n); Q = dQ' * dQ; Q = Q .* (L/max(abs(Q(:))));
    
    Pmax = max(abs(P(:)));
    Kmax = max(abs(K(:)));
    
    
    
    %% Function handles
    fhK = @(x)(K(:, randsample(n,1,true,x)));
    fhKt = @(x)(K(randsample(n,1,true,x), :).');
    fhP = @(x)(P(:, randsample(n,1,true,x)));
    fhQ = @(x)(Q(:, randsample(n,1,true,x)));
    
    fhGapValue = @(x, y)(funGapValuePQK(x, y, P, Q, K, 'mosek'));
    
    fprintf('LipG: %g, LipH: %g, sigma: %g, initial gap: %e\n', Pmax, Kmax, sqrt(8*Pmax + 8*Kmax), fhGapValue(ones(n,1)/n, ones(n,1)/n));
    
    %% Main script
    % Save random stream, so that each algorithm calls exactly the same
    % stochastic oracle
    SRNG = rng;
    
    for MaxIter = [1000, 2000, 5000]
        fprintf('%g iterations.\n', MaxIter);
        %% Commonly used paramters
        par = [];
        par.n = n;
        par.Pmax = Pmax;
        par.Kmax = Kmax;
        par.bVerbose = bVerbose;
        par.MaxIter = MaxIter;
        par.bGapValue = 1;
        par.fhGapValue = fhGapValue;
        par.nu = 1e-16;
        par.Omega = sqrt(2*(1+1e-16/n)*log(n/1e-16+1));
        par.GapEvaluationInterval = MaxIter;
        
        %% SAMP
        fprintf('Running SAMP...\n');
        parSAMP = par;
        parSAMP.LipG = Pmax;
        parSAMP.LipH = Kmax;
        parSAMP.sigma = sqrt(8*Pmax + 8*Kmax);
        GapSAMP = zeros(nRun, 1);
        CPUSAMP = zeros(nRun, 1);
        etcSAMP_all = cell(nRun, 1);
        
        rng(SRNG);
        for i = 1:nRun
            [xSAMP, ySAMP, etcSAMP] = funSAMP(fhP, fhQ, fhK, fhKt, parSAMP);
            GapSAMP(i) = etcSAMP.GapValue(end);
            CPUSAMP(i) = etcSAMP.CPUTime(end);
            etcSAMP_all{i} = etcSAMP;
        end
        fprintf('Without tuning. Mean gap: %e\t std gap: %e\t Mean CPU: %g\n', mean(GapSAMP), std(GapSAMP), mean(CPUSAMP));
        
        %% SMP
        rng(SRNG);
        fprintf('Running SMP...\n');
        parSMP = par;
        parSMP.L = sqrt(2*(Pmax^2 + Kmax^2));
        parSMP.M = 4*sqrt(Pmax^2 + Kmax^2);
        GapSMP = zeros(nRun, 1);
        CPUSMP = zeros(nRun, 1);
        etcSMP_all = cell(nRun, 1);
        
        rng(SRNG);
        for i = 1:nRun
            [xSMP, ySMP, etcSMP] = funSMP(fhP, fhQ, fhK, fhKt, parSMP);
            GapSMP(i) = etcSMP.GapValue(end);
            CPUSMP(i) = etcSMP.CPUTime(end);
            etcSMP_all{i} = etcSMP;
        end
        fprintf('Without tuning. Mean gap: %e\t std gap: %e\t Mean CPU: %g\n', mean(GapSMP), std(GapSMP), mean(CPUSMP));
        
        %% Save results
        if bSave
            diary off;
            diary on;
            save(sprintf('%s/L%d_n%d_N_%d.mat', sResultDir, L, n, MaxIter));
        end
        
    end
end

if bSave
    diary off;
end

