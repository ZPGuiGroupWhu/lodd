import math

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma
from scipy.spatial import ConvexHull

def dcm(
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
    get_knn = NearestNeighbors(n_neighbors=k_num + 1).fit(X).kneighbors(X, return_distance=False)
    get_knn = get_knn[:, 1:]

    angle_var = np.zeros(n)
    if(d==2):
        angle = np.zeros((n, k_num))
        for i in range(n):
            for j in range(k_num):
                delta_x = X[get_knn[i, j], 0] - X[i, 0]
                delta_y = X[get_knn[i, j], 1] - X[i, 1]
                if delta_x == 0:
                    if delta_y == 0:
                        angle[i, j] = 0
                    elif delta_y > 0:
                        angle[i, j] = math.pi / 2
                    else:
                        angle[i, j] = 3 * math.pi / 2
                elif delta_x > 0:
                    if math.atan(delta_y / delta_x) >= 0:
                        angle[i, j] = math.atan(delta_y / delta_x)
                    else:
                        angle[i, j] = 2 * math.pi + math.atan(delta_y / delta_x)
                else:
                    angle[i, j] = math.pi + math.atan(delta_y / delta_x)

        for i in range(n):
            angle_order = sorted(angle[i, :])

            for j in range(k_num - 1):
                point_angle = angle_order[j + 1] - angle_order[j]
                angle_var[i] = angle_var[i] + pow(point_angle - 2 * math.pi / k_num, 2)

            point_angle = angle_order[0] - angle_order[k_num - 1] + 2 * math.pi
            angle_var[i] = angle_var[i] + pow(point_angle - 2 * math.pi / k_num, 2)
            angle_var[i] = angle_var[i] / k_num

        angle_var = angle_var / ((k_num - 1) * 4 * pow(math.pi, 2) / pow(k_num, 2))
    else:
        for i in range(n):
            try:
                dif_x = X[get_knn[i], :] - X[i, :]
                map_x = np.linalg.inv(np.diag(np.sqrt(np.diag(np.dot(dif_x, dif_x.T))))) @ dif_x
                # 计算凸包
                hull = ConvexHull(map_x)
                simplex_num = len(hull.simplices)
                simplex_vol = np.zeros(simplex_num)

                for j in range(simplex_num):
                    simplex_coord = map_x[hull.simplices[j], :]
                    simplex_vol[j] = np.sqrt(np.linalg.det(np.dot(simplex_coord, simplex_coord.T))) / gamma(d)

                angle_var[i] = np.var(simplex_vol)

            except Exception as e:
                print(f"An error occurred at index {i}: {e}")

    id_sorted = np.argsort(angle_var)[::-1]
    bou_id = id_sorted[:int(np.ceil(n * ratio))]
    int_id = np.setdiff1d(np.arange(n), bou_id)

    return int_id, bou_id