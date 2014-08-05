function [Gap, POBJ, DOBJ, SadVal] = funGapValuePQK(x, y, P, Q, K, sMethod)
% Gap = POBJ - DOBJ, where
%   POBJ = max_y .5<Px, x> + <K'y, x> - .5<Qy, y>
%        = -min_y .5<Qy, y> - <Kx, y> - .5<Px, x>
% and
%   DOBJ = min_x .5<Px, x> + <K'y, x> - .5<Qy, y>.

if ~exist('sMethod', 'var')
    sMethod = 'mosek';
end
n = length(x);
% Current value of the saddle function
SadVal = x'*P*x/2 + y'*K*x - y'*Q*y/2;

switch lower(sMethod)
    case 'mosek'
        % --------------------------------------
        % MOSEK
        % --------------------------------------
        param = [];
        param.MSK_DPAR_CHECK_CONVEXITY_REL_TOL = 1e-3;

        A = ones(1, n);
        blc = 1;
        buc = 1;
        blx = zeros(n, 1);
        bux = ones(n, 1);
        % Primal
        res = mskqpopt(Q, -K*x, A, blc, buc, blx, bux, param, 'minimize echo(0)');
        POBJ = x.'*P*x/2 - res.sol.itr.pobjval;
        % Dual
        res = mskqpopt(P, (y.'*K).', A, blc, buc, blx, bux, param, 'minimize echo(0)');
        DOBJ = res.sol.itr.pobjval - y.'*Q*y/2;
end

Gap = POBJ - DOBJ;

end

