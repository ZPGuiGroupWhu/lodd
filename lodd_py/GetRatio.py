import numpy as np

def get_ratio(
        X,
        contri=0.8,
        c=1,
):
    n, dim = X.shape
    if dim < n:
        C = np.cov(X, rowvar=False)
    else:
        C = (1 / n) * (X @ X.T)

    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0

    eigenvalues, _ = np.linalg.eig(C)
    lambda_sorted = np.sort(eigenvalues)[::-1]

    cumsum_lambda = np.cumsum(lambda_sorted)
    sum_lambda = np.sum(lambda_sorted)

    indices = np.where(cumsum_lambda / sum_lambda < contri)[0]

    if indices.size > 0:
        last_index = indices[-1]
    else:
        last_index = -1  # If no indices found, handle accordingly

    # Calculate d
    d = int(np.floor(max(2, min(last_index, np.log2(n))+ 1)))

    ratio = min(0.5, (1 - (n**(1/d) - 2*c**(1/d))**d / n))

    return ratio