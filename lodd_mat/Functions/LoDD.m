function [int_id, bou_id] = LoDD(X, varargin)
%   This function returns the id of internal and boundary points of the N by D matrix X. Each row in X
%   represents an observation.
% 
%   Parameters are:
% 
%   'k_num'      - A non-negative integer specifying the number of nearest neighbors.
%                  Default: 20
%   'ratio'      - A positive scalar specifying the ratio of boundary points.
%                  Default: 0.1
paramNames = {'k_num','ratio'};
defaults   = {20,0.1};
[k_num, ratio] = internal.stats.parseArgs(paramNames, defaults, varargin{:});

[n, d] = size(X);
[get_knn, knn_dis] = knnsearch(X,X,'k',k_num+1);
get_knn(:,1) = [];
knn_dis(:,1) = [];
w = 0.5;
LoDD = zeros(n,1);
for i=1:n
    mapX = (X(get_knn(i,:),:)-X(i,:))./knn_dis(i,:)';
    covMat = cov(mapX)*(k_num-1)/k_num;
    lamda_sum = sum(diag(covMat));
    lamda_sum_2 = sum(diag(covMat*covMat));
    LoDD(i) = w*lamda_sum^2 + (d*(1-w)/(d-1))*((lamda_sum^2-lamda_sum_2));
end

sort_ang = sort(LoDD,'ascend');
lodd_thre = sort_ang(round(n*ratio));
bou_id = find(LoDD <= lodd_thre);
int_id = setdiff(1:n, bou_id);
end