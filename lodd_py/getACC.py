import numpy as np
from scipy.optimize import linear_sum_assignment

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    # Check that the predicted and true labels have the same size
    assert y_pred.size == y_true.size, "Size mismatch between predicted and true labels."

    # Determine the number of unique classes
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    # Create the weight matrix
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Calculate the accuracy
    accuracy = w[row_ind, col_ind].sum() / y_pred.size

    return accuracy