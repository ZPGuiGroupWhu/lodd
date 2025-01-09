![image](https://img.shields.io/badge/MATLAB-R2022a-brightgreen) ![image](https://img.shields.io/badge/Python-3.11-blue) 
# LoDD
We propose a robust and efficient boundary point detection method based on Local Direction Dispersion (LoDD). The core of boundary point detection lies in measuring the difference between boundary points and internal points. It is a common observation that an internal point is surrounded by its neighbors in all directions, while the neighbors of a boundary point tend to be distributed only in a certain directional range. By considering this observation, we adopt density-independent K-Nearest Neighbors (KNN) method to determine neighboring points and design a centrality metric LoDD using the eigenvalues of the covariance matrix to depict the distribution uniformity of KNN. 

# Datasets
| Type | Dataset | Samples | Features | Classes |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Synthetic | DS1 | 459 | 2 | 6 |
| Content Cell  | Content Cell  |

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

> **Python**

Python code of LoDD is in the 'lodd_py' file, where the 'lodd' function provides two parameters for user configuration as follows
```python
def lodd(
        X,
        k_num=20,
        ratio=0.1,
):
    """
        This function returns the id of internal and boundary points of the N by D matrix X. Each row in X
        represents an observation.

        Parameters are:

        'k_num'      - A non-negative integer specifying the number of nearest neighbors.
                       Default: 20
        'ratio'      - A positive scalar specifying the ratio of boundary points.
                       Default: 0.1
    """
```

The 'main1.py' file provides an example for detecting boundary points
```python
import numpy as np
from lodd import lodd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Input data
data = np.loadtxt('Datasets/DS1.txt')

# Obtain data size and true annotations
m = data.shape[1]
X = data[:, :m - 1]
ref = data[:, m - 1]

# Perform the LoDD algorithm
start_time = time.time()
true_ratio = sum(ref)/len(ref)
[int_pts, bou_pts] = lodd(X, k_num=8, ratio=true_ratio)
end_time = time.time()
print("Elapsed time:", end_time - start_time, 's')

# Evaluate the accuracy
res = np.zeros(len(ref))
res[bou_pts] = 1
ACC = accuracy_score(ref, res)
print("Accuracy:", ACC)

# Visualize the result
plt.scatter(X[int_pts, 0], X[int_pts, 1], c='red', s=10, marker='o')
plt.scatter(X[bou_pts, 0], X[bou_pts, 1], c='blue', s=10, marker='o')
plt.show()
```

The 'main2.py' file provides an example for clustering with K-means
```python
from sklearn.preprocessing import MinMaxScaler
from ModKmeans import mod_kmeans
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from getACC import acc

# Input data
raw_data = pd.read_csv('Datasets/Control.csv', header=None)
data = np.array(raw_data)

# Obtain data size and true annotations
m = data.shape[1]
X = data[:, :m - 1]
X = MinMaxScaler().fit(X).transform(X)
ref = data[:, m - 1]
clus_num = len(np.unique(ref))

# Perform the KM+LoDD algorithm
clus = mod_kmeans(X, k_num=20, ratio=0.2, c=clus_num, method='lodd')
ACC = acc(ref, clus)
NMI = normalized_mutual_info_score(ref, clus)
print("Accuracy:", ACC, "NMI:", NMI)
```
