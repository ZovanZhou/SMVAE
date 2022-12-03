import os

# Cancel the warning info from tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import argparse
import numpy as np
from math import exp
from typing import List
import tensorflow as tf
from tqdm import tqdm, trange
from data_loader import DataLoader
from data_parser import DataParser
from data_selector import DataSelector
from model import MVAEEncoder, Decoder
from utils import (
    evaluate4mvae,
    load_model,
    get_one_batch_from_iter,
    train_one_batch4mvae,
    train_one_batch4mvae_labeled,
)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--gamma", default=1.0, type=float)
parser.add_argument("--patience", default=10, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--hidden_dim", default=768, type=int)
parser.add_argument("--latent_dim", default=100, type=int)
parser.add_argument("--n_selected_samples", default=-1, type=int)
parser.add_argument("--weights", default="./weights/model.h5", type=str)
parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])
parser.add_argument("--semi_supervised", default=0, type=int, choices=[0, 1])
parser.add_argument("--mode", choices=["train", "test"], default="train", type=str)
parser.add_argument(
    "--dataset", choices=["twitter2015", "twitter2017"], default="twitter2015", type=str
)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


BERT_PATH = "./pretrain/uncased_L-12_H-768_A-12"

# dataset path
DATASET = {
    "twitter2015": "./dataset/twitter2015",
    "twitter2017": "./dataset/twitter2017",
}
DICT_DATASET = DATASET[args.dataset]

# Semi-supervised Learning Settings
if args.n_selected_samples > 0:
    DataSelector(DICT_DATASET, args.n_selected_samples)

# Parse raw data files
DataParser(DICT_DATASET, args.n_selected_samples)

dataLoader = DataLoader(
    DICT_DATASET,
    BERT_PATH,
    batch_size=args.batch_size,
    n_selected_samples=args.n_selected_samples,
)

LAMB = 1.0 if args.semi_supervised and args.n_selected_samples > 0 else 0
GAMMA = (
    exp(1.0 - args.gamma)
    if args.semi_supervised and args.n_selected_samples > 0
    else 1.0
)
LABEL_SIZE = dataLoader.LABEL_SIZE
NONENTITY_LABEL_IDX = dataLoader.NONENTITY_LABEL_IDX

opt4ae = tf.optimizers.Adam(learning_rate=args.lr)

encoder = MVAEEncoder(BERT_PATH, LABEL_SIZE, args.hidden_dim, args.latent_dim)

dec4txt = Decoder(args.hidden_dim, 768)
dec4img = Decoder(args.hidden_dim, 2048)


def train(
    modules: List,
    optimizer: tf.keras.optimizers.Optimizer,
    dataLoader: DataLoader,
):
    def _train_semisupervised(labeled_data, iter_unlabeled_dataset):
        unlabeled_data = get_one_batch_from_iter(
            iter_unlabeled_dataset, dataLoader.Data("unlabeled")
        )
        loss = train_one_batch4mvae(
            modules,
            optimizer,
            labeled_data,
            unlabeled_data[:-4],
            NONENTITY_LABEL_IDX,
            args.latent_dim,
            LAMB,
            GAMMA,
        )
        return loss

    def _train_supervised(labeled_data, iter_unlabeled_dataset):
        loss = train_one_batch4mvae_labeled(
            modules,
            optimizer,
            labeled_data,
            args.latent_dim,
        )
        return loss

    train_func = (
        _train_semisupervised if args.n_selected_samples > 0 else _train_supervised
    )

    best_results = []
    n_patience = 0
    best_dev_f1 = 0.0
    iter_unlabeled_dataset = (
        iter(dataLoader.Data("unlabeled")) if args.n_selected_samples > 0 else None
    )
    train_size = len(dataLoader.Data("train"))
    for epoch in range(args.epoch):
        losses = []
        loop = tqdm(dataLoader.Data("train"), total=train_size, ncols=80, ascii=True)
        for (
            ind,
            seg,
            f_mask,
            s_mask,
            e_mask,
            s_len,
            e_len,
            img,
            label,
            _,
            _,
            _,
        ) in loop:
            labeled_data = [ind, seg, f_mask, s_mask, e_mask, s_len, e_len, img, label]
            loss = train_func(labeled_data, iter_unlabeled_dataset)
            losses.append(loss)
            loop.set_description_str(f"Epoch [{epoch}/{args.epoch}]")
            loop.set_postfix_str(
                "loss={:^7.5f}".format(
                    np.mean(losses),
                )
            )
        dataLoader.reload_train_data()
        # Whether to save the current model
        save_flag = False
        (dev_prec, dev_rec, dev_f1) = evaluate4mvae(
            modules[0], dataLoader, dtype="dev", output_idx=-1
        )
        print(f"Dev: {dev_prec} {dev_rec} {dev_f1}")
        if dev_f1 > best_dev_f1:
            n_patience = 0
            save_flag = True
            best_dev_f1 = dev_f1
        else:
            n_patience += 1
        if args.patience > 0 and n_patience == args.patience:
            break
        # Save the model
        if save_flag:
            best_results = (dev_prec, dev_rec, dev_f1)
            if args.save_model:
                encoder.save_weights(f"./weights/{args.dataset}.model.h5")

    print(f"Best results: [DEV]:{best_results}")


def test(
    models: List,
    dataLoader: DataLoader,
    optimizer: tf.keras.optimizers.Optimizer,
):
    encoder, dec4txt, dec4img = models
    # Load the saved weights
    load_model(args.weights, models, dataLoader, optimizer, args.latent_dim)
    (dev_prec, dev_rec, dev_f1) = evaluate4mvae(
        encoder, dataLoader, dtype="dev", verbose=True
    )
    print(f"Dev: {dev_prec} {dev_rec} {dev_f1}")
    (test_prec, test_rec, test_f1) = evaluate4mvae(
        encoder, dataLoader, dtype="test", verbose=True
    )
    print(f"Test: {test_prec} {test_rec} {test_f1}")


if __name__ == "__main__":
    if args.mode == "train":
        train(
            [encoder, dec4txt, dec4img],
            opt4ae,
            dataLoader,
        )
    elif args.mode == "test":
        test([encoder, dec4txt, dec4img], dataLoader, opt4ae)
