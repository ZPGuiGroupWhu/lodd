# LoDD
We propose a robust and efficient boundary point detection method based on Local Direction Dispersion (LoDD). The core of boundary point detection lies in measuring the difference between boundary points and internal points. It is a common observation that an internal point is surrounded by its neighbors in all directions, while the neighbors of a boundary point tend to be distributed only in a certain directional range. By considering this observation, we adopt density-independent K-Nearest Neighbors (KNN) method to determine neighboring points and design a centrality metric LoDD using the eigenvalues of the covariance matrix to depict the distribution uniformity of KNN. 
![image](https://github.com/ZPGuiGroupWhu/lodd/blob/main/github.png)

Algorithm Procedure

Given n points, this algorithm first divides all points into |√n| equal bins according to their X values. Then, it assigns numbers to all the points within each bin in ascending order based on the Y coordinate. It generates a grid topological structure by connecting the points with the same numbering along the X direction and connecting the points within each bin along the Y direction in ascending order of the Y values. By arranging all points with a row and column rule, a regular grid composed of unit square cells is constructed.
