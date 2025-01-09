import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from InitialClusCenter import initial_clus_center
from GetRatio import get_ratio
from lodd import lodd

def mod_kmeans(
        X,
        k_num=20,
        ratio=0.1,
        c=1,
        method='lodd',
):
    n = X.shape[0]
    cluster = np.zeros(n)

    if method == 'alodd':
        ratio = get_ratio(X, contri=0.8, c=c)

    int_id, bou_id = lodd(X, k_num=k_num, ratio=ratio)
    C = initial_clus_center(X, c)

    int_clus = KMeans(n_clusters=c, init=C, random_state=42).fit(X[int_id, :]).labels_ + 1

    cluster[int_id] = int_clus
    nbrs = NearestNeighbors(n_neighbors=1).fit(X[int_id, :])
    nearest_int_clus = nbrs.kneighbors(X[bou_id, :], return_distance=False)
    cluster[bou_id] = int_clus[nearest_int_clus.flatten()]

    return cluster