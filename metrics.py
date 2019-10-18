import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import random
nmi = normalized_mutual_info_score
ari = adjusted_rand_score


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
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def inspect_clusters(y_true, y_pred, n_clusters):
    w = np.zeros((1, n_clusters), dtype=np.int64)
    for index in range(len(y_true)):
        if y_true[index] == 0:
            w[0][y_pred[index]] -= 1
        elif y_true[index] == 1:
            w[0][y_pred[index]] += 1
        else:
            #     unsupervised mode
            continue
    for index in range(n_clusters):
        if w[0][index] < 0:
            w[0][index] = 0
        elif w[0][index] > 0:
            w[0][index] = 1
        else:
            if random.random() > 0.5:
                w[0][index] = 1
            else:
                w[0][index] = 0
    labeled = 0
    true_labeled = 0
    for index in range(len(y_true)):
        if y_true[index] == 2:
            continue
        labeled += 1
        if w[0][y_pred[index]] == y_true[index]:
            true_labeled += 1

    return float(true_labeled / labeled), w


# if __name__ == '__main__':
#     y_true = [0, 0, 2, 1, 0, 1, 2, 1, 1, 2]
#     y_pred = [1, 2, 0, 2, 1, 2, 0, 0, 1, 0]
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
#     print(acc(y_true, y_pred))
