% Calculate the prox-mapping under entropy distance generating functions,
% for multi-player game.

% x, g and z are nxk matrices, where n is the dimension of strategy
% portifolio of each player, and k is the number of players.

% For each column i, the output z(:,i) is the solution of 
% argmin_u <g(:,i),u> + sum_j u(j)*log(u(j)/x(j,i).
function z = funGn_ProxMapEntropy(x, g)
    g = reshape(g, size(x));
    g = bsxfun(@minus, g, min(g));
    z = x .* exp(-g);
    s = sum(z);
    z(:, s<=0) = 1/size(x, 1);
    s(s<=0) = 1;
    z = bsxfun(@rdivide, z, s);

end
