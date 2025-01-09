import numpy as np
from scipy.spatial.distance import pdist, squareform


def initial_clus_center(X, k):
    n = X.shape[0]

    D = squareform(pdist(X))

    sort_dist_row = np.sort(pdist(X))
    Tdis = sort_dist_row[int(round((n * (n - 1) / 2) * 0.05))]

    den = np.zeros(n)
    for i in range(n):
        den[i] = np.sum(D[i, :] < Tdis)

    dis = np.zeros(n)
    for i in range(n):
        id_ = np.where(den > den[i])[0]
        if id_.size == 0:
            dis[i] = np.max(D[i, :])
        else:
            dis[i] = np.min(D[i, id_])

    den = (den - np.min(den)) / (np.max(den) - np.min(den))
    dis = (dis - np.min(dis)) / (np.max(dis) - np.min(dis))

    score = 0.5 * den + 0.5 * dis

    id_sorted = np.argsort(score)[::-1]
    C = X[id_sorted[:k], :]

    return C