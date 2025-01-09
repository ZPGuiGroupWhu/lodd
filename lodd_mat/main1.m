% Input data
data = textread('Datasets/DS1.txt');

% Obtain data size and true annotations
n = size(data, 1);
X = data(:, 1:end-1);
ref = data(:, end);

% Perform the LoDD algorithm
addpath Functions\
start_time = clock;
true_ratio = sum(ref)/length(ref);
[int_pts, bou_pts] = LoDD(X, 'k_num', 8, 'ratio', true_ratio);
end_time = clock;
disp(['Elapsed time:', num2str(etime(end_time,start_time)), 's']);

% Evaluate the accuracy
res = zeros(n,1);
res(bou_pts) = 1;
ACC = getACC(ref, res);
disp(['Accuracy:', num2str(ACC)]);

% Visualize the result
plot(X(int_pts,1),X(int_pts,2),'ro');
hold on;
plot(X(bou_pts,1),X(bou_pts,2),'bo');
hold on;
