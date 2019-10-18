import numpy as np
import random


def inspect_clusters(y_true, y_pred, n_clusters):
    w = np.zeros((1, n_clusters), dtype=np.int64)
    for index in range(len(y_true)):
        if y_true[index] == 0:
            w[0][y_pred[index]] -= 1
        elif y_true[index] == 1:
            w[0][y_pred[index]] += 1
        else:
            # unsupervised mode
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

