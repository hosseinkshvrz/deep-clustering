import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # setting the hyper parameters

    parser = argparse.ArgumentParser(description='visualization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filepath', type=str)

    args = parser.parse_args()
    print(args)

    filepath = args.filepath
    embeddings = np.load(filepath)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(embeddings)

    x = principal_components[:, 0]
    y = principal_components[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.show()
