# Grid Generation Algorithm
Given n points, this algorithm first divides all points into |√n| equal bins according to their X values. Then, it assigns numbers to all the points within each bin in ascending order based on the Y coordinate. It generates a grid topological structure by connecting the points with the same numbering along the X direction and connecting the points within each bin along the Y direction in ascending order of the Y values. By arranging all points with a row and column rule, a regular grid composed of unit square cells is constructed.
![image](https://github.com/ZPGuiGroupWhu/lodd/blob/main/lodd_mat/Functions/R4-fig1.png)

# How to Run
This algorithm is implemented in MATLAB and can be used in the 'GenerateGrid' file.
