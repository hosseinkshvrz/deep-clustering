import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


if __name__ == "__main__":
    # setting the hyper parameters

    # parser = argparse.ArgumentParser(description='visualization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--embedding', default='data/embedding_0.npy', type=str)
    # parser.add_argument('--label', default='data/label_0.npy', type=str)
    # parser.add_argument('--xpos', default='x_pos.npy', type=str)
    # parser.add_argument('--ypos', default='y_pos.npy', type=str)
    # parser.add_argument('--xneg', default='x_neg.npy', type=str)
    # parser.add_argument('--yneg', default='y_neg.npy', type=str)
    # parser.add_argument('--xunk', default='x_unk.npy', type=str)
    # parser.add_argument('--yunk', default='y_unk.npy', type=str)
    #
    # args = parser.parse_args()
    # print(args)
    #
    # embedding = args.embedding
    # label = args.label
    # xpos = args.xpos
    # ypos = args.ypos
    # xneg = args.xneg
    # yneg = args.yneg
    # xunk = args.xunk
    # yunk = args.yunk
    #
    # directory = '/home/bsabeti/framework/embeddings/'
    #
    # embeddings = np.load(embedding)
    # labels = np.load(label)
    #
    # pos = []
    # neg = []
    # unk = []
    # for i, data in enumerate(embeddings):
    #     if labels[i] == 0:
    #         neg.append(data)
    #     elif labels[i] == 1:
    #         pos.append(data)
    #     else:
    #         unk.append(data)
    #
    # pos_components = np.asarray(pos)
    # neg_components = np.asarray(neg)
    # unk_components = np.asarray(unk)
    #
    # print('data loaded')
    #
    # # pca = PCA(n_components=2)
    # # pos_components = pca.fit_transform(pos)
    # # neg_components = pca.fit_transform(neg)
    #
    # tsne = TSNE(n_components=2)
    # pos_components = tsne.fit_transform(pos_components)
    # neg_components = tsne.fit_transform(neg_components)
    # unk_components = tsne.fit_transform(unk_components)
    #
    # x_pos = pos_components[:, 0]
    # np.save(directory + xpos, x_pos)
    # y_pos = pos_components[:, 1]
    # np.save(directory + ypos, y_pos)
    #
    # x_neg = neg_components[:, 0]
    # np.save(directory + xneg, x_neg)
    # y_neg = neg_components[:, 1]
    # np.save(directory + yneg, y_neg)
    #
    # x_unk = unk_components[:, 0]
    # np.save(directory + xunk, x_unk)
    # y_unk = unk_components[:, 1]
    # np.save(directory + yunk, y_unk)

    x_pos = np.load('xpos87.npy')
    x_neg = np.load('xneg87.npy')
    y_pos = np.load('ypos87.npy')
    y_neg = np.load('yneg87.npy')
    x_unk = np.load('xunk87.npy')
    y_unk = np.load('yunk87.npy')

    plt.scatter(x_unk, y_unk, color='gray', alpha=0.5)
    plt.scatter(x_pos, y_pos, color='green', alpha=0.5)
    plt.scatter(x_neg, y_neg, color='red', alpha=0.5)
    plt.show()

