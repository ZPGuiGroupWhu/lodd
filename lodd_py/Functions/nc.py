import numpy as np
from sklearn.neighbors import NearestNeighbors

def nc(
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

    # Obtain size of data
    n = X.shape[0]

    # Serach the KNN for each point
    knn_dis, get_knn = NearestNeighbors(n_neighbors=k_num + 1).fit(X).kneighbors(X)
    get_knn = get_knn[:, 1:]
    knn_dis = knn_dis[:, 1:]

    nc = np.zeros(n)
    for i in range(n):
        diff = (X[get_knn[i]] - X[i]) / knn_dis[i, :, np.newaxis]
        S = np.dot(diff, diff.T)
        if np.abs(np.linalg.det(S)) <= np.finfo(float).eps:
            S = S + (0.1 ** 2 / k_num) * np.trace(S) * np.eye(k_num)
        S_inv = np.linalg.inv(S)
        W = (S_inv @ np.ones((k_num, 1))) / (np.ones((1, k_num)) @ S_inv @ np.ones((k_num, 1)))
        nc[get_knn[i]] += ((W > 0) & (W < 1)).flatten()

    sort_ang = np.sort(nc)
    thre = sort_ang[int(np.ceil(n * ratio)) - 1]
    bou_id = np.where(nc <= thre)[0]
    int_id = np.setdiff1d(np.arange(n), bou_id)

    return int_id, bou_id