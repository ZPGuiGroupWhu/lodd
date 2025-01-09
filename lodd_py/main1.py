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