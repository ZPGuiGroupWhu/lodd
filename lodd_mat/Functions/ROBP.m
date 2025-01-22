function [int_id, bou_id] = ROBP(X, k_num, ratio)
n = size(X, 1);
[get_knn, knn_dis] = knnsearch(X,X,'k',k_num+1);
get_knn(:,1) = [];
knn_dis(:,1) = [];
ROBP = zeros(n,1);
for i=1:n
    [rnn, ~] = find(get_knn==i);
    ROBP(i) = sum((pdist2(X(i,:),X(rnn,:))'.^2./knn_dis(rnn,end).^2+1).^(-1));
end
sort_mtc = sort(ROBP,'ascend');
mtc_thre = sort_mtc(ceil(n*ratio));
bou_id = find(ROBP <= mtc_thre);
int_id = setdiff(1:n, bou_id);
end