function [C] = InitialClusCenter(X,k)
[n,~] = size(X);
D = pdist2(X, X);
sortDistRow = sort(pdist(X));
Tdis = sortDistRow(round((n*(n-1)/2)*0.05));
den = zeros(n,1);
for i=1:n
    den(i) = length(find(D(i,:)<Tdis));
end
dis = zeros(n,1);
for i=1:n
    id = find(den>den(i));
    if isempty(id)
        dis(i) = max(pdist2(X(i,:),X));
    else
        dis(i) = min(pdist2(X(i,:),X(id,:)));
    end
end
den = (den-min(den))/(max(den)-min(den));
dis = (dis-min(dis))/(max(dis)-min(dis));
score = 0.5*den+0.5*dis;
[~, id] = sort(score,'descend');
C = X(id(1:k),:);
end

