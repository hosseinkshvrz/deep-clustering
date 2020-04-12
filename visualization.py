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
    # parser.add_argument('--n_labels', type=int)
    # parser.add_argument('--exp', type=str)
    #
    # args = parser.parse_args()
    # print(args)
    #
    # embedding = args.embedding
    # label = args.label
    # n_labels = args.n_labels
    # exp = args.exp
    #
    # directory = '/home/bsabeti/framework/embeddings/'
    #
    # embeddings = np.load(embedding)
    # labels = np.load(label)
    #
    # data = []
    #
    # for i in range(n_labels):
    #     label_ds = []
    #     for j, d in enumerate(embeddings):
    #         if labels[j] == i:
    #             label_ds.append(d)
    #     label_ds = np.asarray(label_ds)
    #     data.append(label_ds)
    #
    # print('data loaded')
    #
    # # pca = PCA(n_components=2)
    # # pos_components = pca.fit_transform(pos)
    # # neg_components = pca.fit_transform(neg)
    #
    # tsne = TSNE(n_components=2)
    # xs = []
    # ys = []
    #
    # for i in range(n_labels):
    #     component = tsne.fit_transform(data[i])
    #     np.save(directory + 'x' + str(exp) + '_' + str(i), component[:, 0])
    #     np.save(directory + 'y' + str(exp) + '_' + str(i), component[:, 1])
    #
    # print('finished')

    # ----------------------------------

    dir = '/home/hossein/Documents/DEC-keras-master/recent/embeddings/'

    x1 = np.load(dir + 'x24_0_0.npy')
    x2 = np.load(dir + 'x24_0_1.npy')
    x3 = np.load(dir + 'x24_0_2.npy')
    # x4 = np.load(dir + 'x13_180_3.npy')
    # x5 = np.load(dir + 'x13_180_4.npy')
    y1 = np.load(dir + 'y24_0_0.npy')
    y2 = np.load(dir + 'y24_0_1.npy')
    y3 = np.load(dir + 'y24_0_2.npy')
    # y4 = np.load(dir + 'y13_180_3.npy')
    # y5 = np.load(dir + 'y13_180_4.npy')

    # plt.scatter(x5, y5, color='gray', alpha=0.3)
    plt.scatter(x1, y1, color='red', alpha=0.3)
    # plt.scatter(x3, y3, color='blue', alpha=0.3)
    # plt.scatter(x4, y4, color='yellow', alpha=0.3)
    plt.scatter(x2, y2, color='green', alpha=0.3)
    plt.savefig('figures/24'
                ''
                '_0.png')
    # plt.show()

