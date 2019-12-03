import numpy as np
import random


# def inspect_clusters(y_true, y_pred, n_clusters):
#     w = np.zeros((1, n_clusters), dtype=np.int64)
#     for index in range(len(y_true)):
#         if y_true[index] == 0:
#             w[0][y_pred[index]] -= 1
#         elif y_true[index] == 1:
#             w[0][y_pred[index]] += 1
#         else:
#             # unsupervised mode
#             continue
#     for index in range(n_clusters):
#         if w[0][index] < 0:
#             w[0][index] = 0
#         elif w[0][index] > 0:
#             w[0][index] = 1
#         else:
#             if random.random() > 0.5:
#                 w[0][index] = 1
#             else:
#                 w[0][index] = 0
#     labeled = 0
#     true_labeled = 0
#     for index in range(len(y_true)):
#         if y_true[index] == 2:
#             continue
#         labeled += 1
#         if w[0][y_pred[index]] == y_true[index]:
#             true_labeled += 1
#
#     return float(true_labeled / labeled), w

def inspect_clusters(y_true, y_pred, n_clusters):
    w = np.zeros((4, n_clusters), dtype=np.int64)
    weight = np.zeros((4, n_clusters), dtype=np.int64)
    for index in range(len(y_true)):
        w[y_true[index]][y_pred[index]] += 1

    print(w)

    # order = [1, 2, 3, 0]
    order = [0, 2, 3, 1]
    for i in order:
        cluster_index = w[i].argmax()
        tmp = np.zeros(n_clusters, dtype='int64')
        tmp[cluster_index] = 1
        weight[i] = tmp
        w[:, cluster_index] = 0
        print(w)
        print(weight)

    labeled = 0
    true_labeled = 0
    for index in range(len(y_true)):
        labeled += 1
        if weight[y_true[index]][y_pred[index]] == 1:
            true_labeled += 1

    return float(true_labeled / labeled), weight
