![image](https://img.shields.io/badge/MATLAB-R2022a-brightgreen) ![image](https://img.shields.io/badge/Python-3.11-blue) 
# LoDD
We propose a robust and efficient boundary point detection method based on Local Direction Dispersion (LoDD). The core of boundary point detection lies in measuring the difference between boundary points and internal points. It is a common observation that an internal point is surrounded by its neighbors in all directions, while the neighbors of a boundary point tend to be distributed only in a certain directional range. By considering this observation, we adopt density-independent K-Nearest Neighbors (KNN) method to determine neighboring points and design a centrality metric LoDD using the eigenvalues of the covariance matrix to depict the distribution uniformity of KNN. 

# How To Run
> **MATLAB**

MATLAB code of LoDD is in the 'lodd_mat' file, where the 'lodd' function provides multiple hyperparameters for user configuration as follows 
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
