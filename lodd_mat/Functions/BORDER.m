function [int_id, bou_id] = BORDER(X, k_num, ratio)
    n = size(X,1);
    [get_knn, ~] = knnsearch(X,X,'k',k_num+1);
    get_knn(:,1) = [];
    count = tabulate(get_knn(:));
    rnn = count(:,2);
    [~, bou_id] = mink(rnn, round(n*ratio));
    int_id = setdiff(1:n, bou_id);
end