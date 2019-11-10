import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


if __name__ == "__main__":
    # setting the hyper parameters

    parser = argparse.ArgumentParser(description='visualization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filepath', default='data/embedding_19999.npy', type=str)

    args = parser.parse_args()
    print(args)

    filepath = args.filepath
    embeddings = np.load(filepath)
    labels = np.load('data/label_0.npy')

    pos = []
    neg = []
    for i, data in enumerate(embeddings):
        if labels[i] == 0:
            neg.append(data)
        elif labels[i] == 1:
            pos.append(data)

    pos = np.asarray(pos)
    neg = np.asarray(neg)

    pca = PCA(n_components=2)
    pos_components = pca.fit_transform(pos)
    neg_components = pca.fit_transform(neg)

    # tsne = TSNE(n_components=2)
    # pos_components = tsne.fit_transform(pos_components)
    # neg_components = tsne.fit_transform(neg_components)

    x_pos = pos_components[:, 0]
    y_pos = pos_components[:, 1]

    x_neg = neg_components[:, 0]
    y_neg = neg_components[:, 1]

    # fig, ax = plt.subplots()
    # ax.scatter(x, y)
    plt.scatter(x_pos, y_pos, color='green')
    plt.scatter(x_neg, y_neg, color='red')
    plt.show()
