import ast
import os
import argparse
from time import time
from comet_ml import Experiment

import batch_datasets
from batch_model import DSC

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


experiment = Experiment(api_key="3UuCR8Zz4hqHn9aM9bkR1jXdr",
                        project_name="dec", workspace="hossein-kshvrz")

if __name__ == "__main__":
    # setting the hyper parameters

    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='IMDB', choices=['IMDB', 'SST', 'Reuters'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_iter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=10, type=int)
    parser.add_argument('--update_interval', default=200, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--n_clusters', default=100, type=int)
    parser.add_argument('--latent_dims', default='[128]', type=str)
    parser.add_argument('--ae_type', default='lstm_ae', type=str)
    parser.add_argument('--train_mode', default='semi-supervised', choices=['supervised', 'semi-supervised'])
    parser.add_argument('--mask_label', default=0.0, type=float)
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
    latent_dims = ast.literal_eval(args.latent_dims)
    ae_type = args.ae_type
    train_mode = args.train_mode
    mask_label = args.mask_label

    module = batch_datasets
    dataset_class = getattr(module, dataset)
    dataset_obj = dataset_class(train_mode)
    directory, train_files, valid_file, labels, doc_dims, mask_file = dataset_obj.get_data()

    dsc = DSC(directory=directory,
              train_files=train_files,
              valid_file=valid_file,
              labels=labels,
              doc_dims=doc_dims,
              latent_dims=latent_dims,
              ae_type=ae_type,
              n_clusters=n_clusters,
              mask_label=mask_label,
              mask_file=mask_file)

    dsc.pretrain(epochs=pretrain_epochs,
                 batch_size=batch_size,
                 save_dir=save_dir)

    dsc.compile(loss='kld')
    dsc.model.summary()

    t0 = time()

    dsc.fit(max_iter=max_iter,
            batch_size=batch_size,
            update_interval=update_interval,
            save_dir=save_dir)

    print('clustering time: ', (time() - t0))
