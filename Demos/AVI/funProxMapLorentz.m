function rnew = funProxMapLorentz(r, h)
% Solve the problem 
%   u = argmin ||u-(r-h)||^2
% where u is in the Lorentz cone.

    rnew = r - h;
    normr = norm(rnew(1:end-1));
    if normr <= -rnew(end)
        rnew = zeros(size(r));
    elseif normr > abs(rnew(end))
        tmp = .5*(1 + rnew(end)/normr);
        rnew(end) = normr;
        rnew = rnew*tmp;
    end
end