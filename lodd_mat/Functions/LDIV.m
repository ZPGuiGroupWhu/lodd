function [int_id, bou_id] = LDIV(X, k_num, ratio)
[n, ~] = size(X);
[get_knn, ~] = knnsearch(X,X,'k',k_num+1);
get_knn(:,1) = [];
count = tabulate(get_knn(:));
rnn = count(:,2);

LDIV = zeros(n,1);
for i=1:n
    dis = pdist2(X(get_knn(i,:),:),X(get_knn(i,:),:));
    mid = find(min(sum(dis))==sum(dis),1);
    medoid = X(get_knn(i,mid),:);
    LDIV(i) = pdist2(X(i,:),medoid)./sum(pdist2(X(get_knn(i,:),:),medoid).*rnn(get_knn(i,:)));
end
[~, bou_id] = maxk(LDIV, round(n*ratio));
int_id = setdiff(1:n, bou_id);
end