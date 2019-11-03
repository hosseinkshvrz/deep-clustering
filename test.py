import os
import argparse
from time import time
import datasets
import numpy as np
from keras.models import load_model
from metrics import inspect_clusters
from model import ClusteringLayer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # setting the hyper parameters

    parser = argparse.ArgumentParser(description='test', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='IMDB', choices=['IMDB', 'SST'])
    parser.add_argument('--filepath', default='results/model.h5')
    parser.add_argument('--n_clusters', default=100)
    parser.add_argument('--train_mode', default='semi-supervised', choices=['supervised', 'semi-supervised'])

    args = parser.parse_args()

    print(args)

    dataset = args.dataset
    filepath = args.filepath
    train_mode = args.train_mode
    n_clusters = args.n_clusters

    model = load_model(filepath, custom_objects={'ClusteringLayer': ClusteringLayer})
    model.summary()

    # strangely doesn't work in my local system, but works on GPU server
    module = datasets
    dataset_class = getattr(module, dataset)
    dataset_obj = dataset_class(train_mode)
    x_test, y_test = dataset_obj.get_test_data()

    x_test = x_test.astype(np.float16)
    y_test = y_test.astype(np.float16)

    q_test = model.predict(x_test)
    y_pred = q_test.argmax(1)
    acc, _ = inspect_clusters(y_test, y_pred, n_clusters)
    print(acc)
