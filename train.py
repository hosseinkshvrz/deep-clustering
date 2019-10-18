import os
import argparse
from time import time
from sklearn.utils import shuffle
from datasets import load_data
from recent.model import DEC

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    # setting the hyper parameters

    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='imdb', choices=['imdb', 'sst', 'amazon'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_iter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=10, type=int)
    parser.add_argument('--update_interval', default=200, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--save_dir', default='results')
    parser.add_argument('--n_clusters', default=100)
    parser.add_argument('--latent_dims', default=[128], type=list)
    parser.add_argument('--ae_type', default='lstm_ae', type=str)
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = args.dataset
    batch_size = args.batch_size
    max_iter = args.max_iter
    pretrain_epochs = args.pretrain_epochs
    update_interval = args.update_interval
    tol = args.tol
    save_dir = args.save_dir
    n_clusters = args.n_clusters
    latent_dims = args.latent_dims
    ae_type = args.ae_type

    x, y, x_test, y_test = load_data(dataset)

    doc_dims = x.shape[1:]

    x, y = shuffle(x, y)
    x_test, y_test = shuffle(x_test, y_test)

    train_sample_size = int(len(x) * 0.8)
    x_train = x[:train_sample_size]
    y_train = y[:train_sample_size]
    x_valid = x[train_sample_size:]
    y_valid = y[train_sample_size:]

    dec = DEC(doc_dims=doc_dims, latent_dims=latent_dims, ae_type=ae_type, n_clusters=n_clusters)
    dec.pretrain(x=x,
                 y=y,
                 x_valid=x_valid,
                 y_valid=y_valid,
                 epochs=pretrain_epochs,
                 batch_size=batch_size,
                 save_dir=save_dir)

    dec.compile(loss='kld')
    dec.model.summary()

    t0 = time()
    dec.fit(x=x,
            y=y,
            x_valid=x_valid,
            y_valid=y_valid,
            tol=tol,
            maxiter=max_iter,
            batch_size=batch_size,
            update_interval=update_interval, save_dir=save_dir)

    print('clustering time: ', (time() - t0))
