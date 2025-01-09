function ratio = GetRatio(X, varargin)
paramNames = {'Contribution','NumClus'};
defaults   = {0.8, 1};
[contri, c] = internal.stats.parseArgs(paramNames, defaults, varargin{:});
[n, dim] = size(X);
if dim < n
    C = cov(X);
else
    C = (1 / n) * (X * X');
end
C(isnan(C)) = 0;
C(isinf(C)) = 0;
[~, lambda] = eig(C);
lambda = sort(diag(lambda),'descend');
d = floor(max(2, min(find(cumsum(lambda)/sum(lambda) < contri, 1, 'last' ),log2(n))));
ratio = min(0.5, 1 - (n^(1/d)-2*c^(1/d))^d/n);
end