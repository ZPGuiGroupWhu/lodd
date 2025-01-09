![image](https://img.shields.io/badge/MATLAB-R2022a-brightgreen) ![image](https://img.shields.io/badge/Python-3.11-blue) 
# LoDD
We propose a robust and efficient boundary point detection method based on Local Direction Dispersion (LoDD). The core of boundary point detection lies in measuring the difference between boundary points and internal points. It is a common observation that an internal point is surrounded by its neighbors in all directions, while the neighbors of a boundary point tend to be distributed only in a certain directional range. By considering this observation, we adopt density-independent K-Nearest Neighbors (KNN) method to determine neighboring points and design a centrality metric LoDD using the eigenvalues of the covariance matrix to depict the distribution uniformity of KNN. 

# How To Run
> **MATLAB**

MATLAB code of LoDD is in the 'lodd_mat' file, where the 'lodd' function provides two parameters for user configuration as follows 
```matlab
function [int_id, bou_id] = LoDD(X, varargin)
%   This function returns the id of internal and boundary points of the N by D matrix X. Each row in X represents an observation.
% 
%   Parameters are:
% 
%   'k_num'      - A non-negative integer specifying the number of nearest neighbors.
%                  Default: 20
%   'ratio'      - A positive scalar specifying the ratio of boundary points.
%                  Default: 0.1
```

The 'main1.m' file provides an example for detecting boundary points
```matlab
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
```

The 'main2.m' file provides an example for clustering with K-means
```matlab
% Input data
data = csvread('Datasets/Control.csv');

% Obtain data size and true annotations
m = size(data, 2);
X = data(:,1:m-1);
X = mapminmax(X',0,1)';
ref = data(:,m);
clus_num = length(unique(ref));

% Perform the KM+LoDD algorithm
clus = ModKmeans(X,'k_num',20,'ratio',0.2,'NumClus',clus_num,'Method','lodd');
ACC = getACC(ref,clus);
NMI = getNMI(ref,clus);
disp(['ACC:', num2str(ACC), ' NMI:', num2str(NMI)]);
```
