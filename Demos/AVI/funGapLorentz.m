function GapValue = funGapLorentz(B, C, b, w, v)
%   Evaluate
%       g(w,v) := sup_u G(w)-G(u) + <H(u)-v, w-u>
%   among all u in the Lorentz cone, where
%       G(u) = |Bu|^2/2 + <b,u>, 
%   and
%       H(u) = C*u.
%   In fact, g(w,v) is the gap function for the affine variational
%   inequality problem, which computes w* such that
%       <Au+b, w*-u> <= 0, for all u in the Lorentz cone,
%   where A is a linear monotone operator. (Hence A can be
%   decomposed to the sum of a p.s.d matrix B'B and a skew-symmetric matrix
%   C).

%   The problem can be formulated as
%           G(w) - <v, w> + sup_u -G(u)-<H(u),u> + <H(u), w>+<v,u>
%       =   G(w) - <v, w> - inf_u G(u) - <H(u), w> - <v, u>
%       =   G(w) - <v, w> - inf_u |Bu|^2/2 + <b + C*w - v, u>


    [m, n] = size(B);
    f = b + C*w - v;
    const = norm(B*w)^2/2 + (b-v).'*w;
    
    % MOSEK
    p = [];
    [~, res] = mosekopt('symbcon echo(0)');
    p.c = [f.', zeros(1, m), 0, 1];
    p.a = sparse([B, -eye(m), zeros(m, 2)]);
    p.blc = zeros(1, m);
    p.buc = p.blc;
    p.blx = [-inf(1, n), -inf(1, m), 1, -inf];
    p.bux = [inf(1, n), inf(1, m), 1, inf];
    p.cones.type = [res.symbcon.MSK_CT_QUAD, res.symbcon.MSK_CT_RQUAD];
    p.cones.sub = [n, 1:(n-1), n+m+1, n+m+2, (n+1):(n+m)];
    p.cones.subptr = [1, n+1];
    [~, res] = mosekopt('minimize echo(0)', p);
    
    GapValue = const - res.sol.itr.pobjval;
end