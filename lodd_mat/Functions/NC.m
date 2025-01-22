function [int_id, bou_id] = NC(X, k_num, ratio)
n = size(X,1);
[get_knn, knn_dis] = knnsearch(X,X,'k',k_num+1);
get_knn(:,1) = [];
knn_dis(:,1) = [];
NC = zeros(n,1);
for i=1:n
    S = ((X(get_knn(i,:),:)-X(i,:))./knn_dis(i,:)')*((X(get_knn(i,:),:)-X(i,:))./knn_dis(i,:)')';
    W = (inv(S)*ones(k_num,1))/(ones(1,k_num)*inv(S)*ones(k_num,1));
    NC(get_knn(i,:)) = NC(get_knn(i,:)) + (W>0 & W<1);
end

sort_mtc = sort(NC,'ascend');
mtc_thre = sort_mtc(ceil(n*ratio));
bou_id = find(NC <= mtc_thre);
int_id = setdiff(1:n, bou_id);
end