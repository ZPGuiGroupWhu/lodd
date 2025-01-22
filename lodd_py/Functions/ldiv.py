import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

def ldiv(
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
    get_knn = NearestNeighbors(n_neighbors=k_num + 1).fit(X).kneighbors(X, return_distance=False)
    get_knn = get_knn[:, 1:]
    unique, counts = np.unique(get_knn, return_counts=True)
    rnn = np.zeros(n)
    rnn[unique] = counts

    LDIV = np.zeros(n)
    for i in range(n):
        dis = cdist(X[get_knn[i]], X[get_knn[i]])
        mid = np.argmin(np.sum(dis, axis=1))
        medoid = X[get_knn[i][mid]]  # 获取该点
        distance_to_medoid = cdist([X[i]], [medoid])[0][0]
        distances_to_neighbors = cdist(X[get_knn[i]], [medoid]).flatten()
        LDIV[i] = distance_to_medoid / np.sum(distances_to_neighbors * rnn[get_knn[i]])

    id_sorted = np.argsort(LDIV)[::-1]
    bou_id = id_sorted[:int(np.ceil(n * ratio))]
    int_id = np.setdiff1d(np.arange(n), bou_id)

    return int_id, bou_id