function [GapValue, u] = funGn_GapValue(B, C, w)
%   Evaluate
%       g(w) := sup_u G(w)-G(u) + <H(u), w-u>
%   among all u in Z, where Z is k products of n-simplices
%       G(u) = |Bu|^2/2, 
%   and
%       H(u) = C*u.
%   In fact, g(w) is the gap function for the affine variational
%   inequality problem, which computes w* in Z such that
%       <Au, w*-u> <= 0, for all u in Z,
%   where A is a linear monotone operator. (Hence A can be
%   decomposed to the sum of a p.s.d matrix B'B=(A+A')/2 and a 
%   skew-symmetric matrix C=(A-A')/2).

%   The problem can be formulated as
%           G(w) + sup_u -G(u)-<H(u),u> + <H(u), w>
%       =   G(w) - inf_u G(u) - <H(u), w>
%       =   G(w) - inf_u |Bu|^2/2 + <C*w, u>

    [n, k] = size(w);
    km = size(B, 1);
    w = w(:);
    
    % MOSEK
    p = [];
    [~, res] = mosekopt('symbcon echo(0)');
    p.c = [C*w; zeros(km,1); 0; 1].';
    p.a = sparse([B, -eye(km), zeros(km, 2);  kron(eye(k,k), ones(1,n)), zeros(k, km+2)]);
    p.blc = [zeros(1, km), ones(1, k)];
    p.buc = p.blc;
    p.blx = [zeros(1, k*n), -inf(1, km), 1, -inf];
    p.bux = [ones(1, k*n), inf(1, km), 1, inf];
    p.cones.type = [res.symbcon.MSK_CT_RQUAD];
    p.cones.sub = [k*n+km+1, k*n+km+2, (k*n+1):(k*n+km)];
    p.cones.subptr = 1;
    param = [];
    param.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1.0e-15;
    param.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1.0e-15;
    param.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1.0e-15;

    [~, res] = mosekopt('minimize echo(0)', p, param);
    
    GapValue = norm(B*w)^2/2 - res.sol.itr.pobjval;
    u = res.sol.itr.xx(1:(k*n));

    % --------------
    % DEBUG
    % --------------
    test = norm(B*w)^2/2 - norm(B*u)^2/2  + (C*u).'*(w-u);
    if abs(test-GapValue)>1e-3
        error('wrong! |%g-%g|=%g', test, GapValue, abs(test-GapValue));
    end
    
    u = reshape(u, [n, k]);
end