import numpy as np
from sklearn.neighbors import NearestNeighbors

def lodd(
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

    # Obtain size and dimension of data
    n, d = X.shape

    # Serach the KNN for each point
    knn_dis, get_knn = NearestNeighbors(n_neighbors=k_num + 1).fit(X).kneighbors(X)
    get_knn = get_knn[:, 1:]
    knn_dis = knn_dis[:, 1:]

    w = 0.5
    LoDD = np.zeros(n)
    for i in range(n):
        knn_pts = X[get_knn[i], :] - X[i]
        mapX = knn_pts / knn_dis[i][:, np.newaxis]
        covMat = np.cov(mapX, rowvar=False) * (k_num - 1) / k_num
        lamda_sum = np.sum(np.diag(covMat))
        lamda_sum_2 = np.sum(np.diag(np.dot(covMat, covMat)))
        LoDD[i] = w * lamda_sum ** 2 + (d * (1 - w) / (d - 1)) * (lamda_sum ** 2 - lamda_sum_2)

    sort_ang = np.sort(LoDD)
    lodd_thre = sort_ang[int(np.ceil(n * ratio)) - 1]
    bou_id = np.where(LoDD <= lodd_thre)[0]
    int_id = np.setdiff1d(np.arange(n), bou_id)

    return int_id, bou_id