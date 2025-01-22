import numpy as np
from sklearn.neighbors import NearestNeighbors

def border(
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

    count = np.unique(get_knn, return_counts=True)
    rnn = count[1]

    bou_id = np.argsort(rnn)[:int(round(n * ratio))]
    int_id = np.setdiff1d(np.arange(n), bou_id)

    return int_id, bou_id