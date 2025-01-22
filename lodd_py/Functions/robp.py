import numpy as np
from sklearn.neighbors import NearestNeighbors

def robp(
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

    ROBP = np.zeros(n)
    for i in range(n):
        rnn = np.where(get_knn == i)[0]
        dis = np.linalg.norm(X[rnn] - X[i], axis=1)
        ROBP[i] = np.sum((dis ** 2 / knn_dis[rnn, -1] ** 2 + 1) ** (-1))

    sort_ang = np.sort(ROBP)
    thre = sort_ang[int(np.ceil(n * ratio)) - 1]
    bou_id = np.where(ROBP <= thre)[0]
    int_id = np.setdiff1d(np.arange(n), bou_id)

    return int_id, bou_id