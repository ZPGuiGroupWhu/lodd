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
