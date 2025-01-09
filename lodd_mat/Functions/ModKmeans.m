function cluster = ModKmeans(X, varargin)
paramNames = {'k_num','ratio','NumClus','Method'};
defaults   = {20,0.1,1,'lodd'};
[k_num, ratio, c, method] = internal.stats.parseArgs(paramNames, defaults, varargin{:});
[n, ~] = size(X);
cluster = zeros(n,1);
if(strcmp(method, 'alodd'))
    ratio = GetRatio(X, 'Contribution', 0.8, 'NumClus', c);
end
[int_id, bou_id] = LoDD(X, 'k_num', k_num, 'ratio', ratio);
C = InitialClusCenter(X,c);
int_clus = kmeans(X(int_id,:),c,"Start",C);
cluster(int_id) = int_clus;
cluster(bou_id) = int_clus(knnsearch(X(int_id,:),X(bou_id,:),'k',1));
end