import os
import logging
import numpy as np
from tqdm import tqdm
from typing import List
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from conlleval import evaluate as evaluate_ner


# Logger config
def getLogger(name: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def get_one_batch_from_iter(iter_dataset, dataset):
    batch_data = []
    try:
        batch_data = next(iter_dataset)
    except Exception:
        iter_dataset = iter(dataset)
        batch_data = next(iter_dataset)
    return batch_data


@tf.function
def train_one_batch4mvae(
    modules: List,
    optimizer: tf.keras.optimizers.Optimizer,
    labeled_data: List,
    unlabeled_data: List,
    NONENTITY_LABEL_IDX: int,
    latent_dim: int,
    lamb: float,
    gamma: float,
):
    enc, dec4txt, dec4img = modules
    x_l, y_l = labeled_data[:-1], labeled_data[-1]
    x_u = unlabeled_data
    batch_size = y_l.get_shape()[0]

    with tf.GradientTape() as tape:
        # labeled data
        (
            span_feature_l,
            txt_mu,
            txt_logvar,
            img_feature_l,
            img_mu,
            img_logvar,
            fusion_mu,
            fusion_logvar,
            logits,
        ) = enc(*x_l)

        sampled_normal = tf.random.normal((batch_size, latent_dim))

        sampling_z4txt = txt_mu + tf.math.exp(txt_logvar / 2) * sampled_normal
        recon_span_feature_l = dec4txt(tf.concat([logits, sampling_z4txt], axis=-1))
        recon_loss_l4txt = tf.losses.mean_squared_error(
            span_feature_l, recon_span_feature_l
        )
        kl_loss_l4txt = -0.5 * tf.reduce_sum(
            1.0 + txt_logvar - tf.math.square(txt_mu) - tf.math.exp(txt_logvar), axis=-1
        )

        sampling_z4img = img_mu + tf.math.exp(img_logvar / 2) * sampled_normal
        recon_img_feature_l = dec4img(sampling_z4img)
        recon_loss_l4img = tf.losses.mean_squared_error(
            img_feature_l, recon_img_feature_l
        )
        kl_loss_l4img = -0.5 * tf.reduce_sum(
            1.0 + img_logvar - tf.math.square(img_mu) - tf.math.exp(img_logvar), axis=-1
        )

        kl_loss_l4fusion = -0.5 * tf.reduce_sum(
            1.0
            + fusion_logvar
            - tf.math.square(fusion_mu)
            - tf.math.exp(fusion_logvar),
            axis=-1,
        )

        supervised_loss = tf.reduce_mean(
            tf.losses.categorical_crossentropy(y_l, logits)
        )
        labeled_data_loss = tf.reduce_mean(
            recon_loss_l4txt
            + recon_loss_l4img
            + kl_loss_l4txt
            + kl_loss_l4img
            + kl_loss_l4fusion
        )

        # unlabeled data
        (
            span_feature_u,
            txt_mu,
            txt_logvar,
            img_feature_u,
            img_mu,
            img_logvar,
            fusion_mu,
            fusion_logvar,
            logits,
        ) = enc(*x_u)

        sampling_z4txt = txt_mu + tf.math.exp(txt_logvar / 2) * sampled_normal

        # original
        recon_span_feature_u = dec4txt(tf.concat([logits, sampling_z4txt], axis=-1))
        recon_loss_u4txt = tf.losses.mean_squared_error(
            span_feature_u, recon_span_feature_u
        )
        kl_loss_u4txt = -0.5 * tf.reduce_sum(
            1.0 + txt_logvar - tf.math.square(txt_mu) - tf.math.exp(txt_logvar), axis=-1
        )

        sampling_z4img = img_mu + tf.math.exp(img_logvar / 2) * sampled_normal

        # original
        recon_img_feature_u = dec4img(sampling_z4img)
        recon_loss_u4img = tf.losses.mean_squared_error(
            img_feature_u, recon_img_feature_u
        )
        kl_loss_u4img = -0.5 * tf.reduce_sum(
            1.0 + img_logvar - tf.math.square(img_mu) - tf.math.exp(img_logvar), axis=-1
        )

        kl_loss_u4fusion = -0.5 * tf.reduce_sum(
            1.0
            + fusion_logvar
            - tf.math.square(fusion_mu)
            - tf.math.exp(fusion_logvar),
            axis=-1,
        )

        mask = tf.where(
            tf.argmax(logits, axis=-1) != NONENTITY_LABEL_IDX,
            x=tf.ones((batch_size,)),
            y=tf.zeros((batch_size,)),
        )
        unlabeled_data_loss = tf.reduce_mean(
            mask
            * (
                recon_loss_u4txt
                + recon_loss_u4img
                + kl_loss_u4txt
                + kl_loss_u4img
                + kl_loss_u4fusion
            )
        )
        model_loss = (
            gamma * supervised_loss + labeled_data_loss + lamb * unlabeled_data_loss
        )
    grads = tape.gradient(
        model_loss,
        enc.trainable_variables
        + dec4txt.trainable_variables
        + dec4img.trainable_variables,
    )
    optimizer.apply_gradients(
        zip(
            grads,
            enc.trainable_variables
            + dec4txt.trainable_variables
            + dec4img.trainable_variables,
        )
    )
    return model_loss


# one step for back-propagation
@tf.function
def train_one_batch4mvae_labeled(
    modules: List,
    optimizer: tf.keras.optimizers.Optimizer,
    labeled_data: List,
    latent_dim: int,
):
    enc, dec4txt, dec4img = modules
    x_l, y_l = labeled_data[:-1], labeled_data[-1]
    batch_size = y_l.get_shape()[0]

    with tf.GradientTape() as tape:
        # labeled data
        (
            span_feature_l,
            txt_mu,
            txt_logvar,
            img_feature_l,
            img_mu,
            img_logvar,
            fusion_mu,
            fusion_logvar,
            logits,
        ) = enc(*x_l)

        sampled_normal = tf.random.normal((batch_size, latent_dim))

        sampling_z4txt = txt_mu + tf.math.exp(txt_logvar / 2) * sampled_normal
        recon_span_feature_l = dec4txt(tf.concat([logits, sampling_z4txt], axis=-1))
        recon_loss_l4txt = tf.losses.mean_squared_error(
            span_feature_l, recon_span_feature_l
        )
        kl_loss_l4txt = -0.5 * tf.reduce_sum(
            1.0 + txt_logvar - tf.math.square(txt_mu) - tf.math.exp(txt_logvar), axis=-1
        )

        sampling_z4img = img_mu + tf.math.exp(img_logvar / 2) * sampled_normal
        recon_img_feature_l = dec4img(sampling_z4img)
        recon_loss_l4img = tf.losses.mean_squared_error(
            img_feature_l, recon_img_feature_l
        )
        kl_loss_l4img = -0.5 * tf.reduce_sum(
            1.0 + img_logvar - tf.math.square(img_mu) - tf.math.exp(img_logvar), axis=-1
        )
        kl_loss_l4fusion = -0.5 * tf.reduce_sum(
            1.0
            + fusion_logvar
            - tf.math.square(fusion_mu)
            - tf.math.exp(fusion_logvar),
            axis=-1,
        )

        supervised_loss = tf.losses.categorical_crossentropy(y_l, logits)
        labeled_data_loss = tf.reduce_mean(
            recon_loss_l4txt
            + recon_loss_l4img
            + kl_loss_l4txt
            + kl_loss_l4img
            + kl_loss_l4fusion
            + supervised_loss
        )
    grads = tape.gradient(
        labeled_data_loss,
        enc.trainable_variables
        + dec4txt.trainable_variables
        + dec4img.trainable_variables,
    )
    optimizer.apply_gradients(
        zip(
            grads,
            enc.trainable_variables
            + dec4txt.trainable_variables
            + dec4img.trainable_variables,
        )
    )
    return labeled_data_loss


def _collect_data(sent_ids, sent_lens, range_ids, probs, preds, labels):
    samples = {}
    for i in range(len(sent_ids)):
        sent_id = sent_ids[i]
        range_id = range_ids[i]
        if sent_id not in samples:
            samples[sent_id] = {"p": [], "t": [], "len": sent_lens[i]}
        if preds[i] != "O":
            samples[sent_id]["p"].append((range_id[0], range_id[1], preds[i], probs[i]))
        if labels[i] != "O":
            samples[sent_id]["t"].append((range_id[0], range_id[1], labels[i]))
    return samples


def _filter_data(samples):
    for sent_id in samples.keys():
        segments = samples[sent_id]["p"]
        ordered_seg = sorted(segments, key=lambda e: -e[-1])
        filter_list = []
        for elem in ordered_seg:
            flag = False
            current = (elem[0], elem[1])
            for prior in filter_list:
                flag = conflict_judge(current, (prior[0], prior[1]))
                if flag:
                    break
            if not flag:
                filter_list.append((elem[0], elem[1], elem[2]))
        samples[sent_id]["p"] = sorted(filter_list, key=lambda e: e[0])
    return samples


def conflict_judge(line_x, line_y):
    if line_x[0] == line_y[0]:
        return True
    if line_x[0] < line_y[0]:
        if line_x[1] >= line_y[0]:
            return True
    if line_x[0] > line_y[0]:
        if line_x[0] <= line_y[1]:
            return True
    return False


def iob_tagging(entities, s_len):
    tags = ["O"] * s_len

    for el, er, et in entities:
        for i in range(el, er + 1):
            if i == el:
                tags[i] = "B-" + et
            else:
                tags[i] = "I-" + et
    return tags


def evaluate4mvae(
    model: tf.keras.models.Model,
    dataLoader,
    dtype: str = "dev",
    output_idx: int = -1,
    verbose: bool = False,
) -> None:
    data = dataLoader.Data(dtype)
    sent_ids, range_ids = [], []
    labels, logits = [], []
    sent_lens = []
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
        sent_id,
        sent_len,
        range_id,
    ) in tqdm(data, ascii=True, ncols=80):
        logit = model(
            ind, seg, f_mask, s_mask, e_mask, s_len, e_len, img, training=False
        )[output_idx]
        labels.append(label)
        logits.append(logit)
        sent_ids.append(sent_id)
        sent_lens.append(sent_len)
        range_ids.append(range_id)
    probs = tf.reduce_max(tf.concat(logits, axis=0), axis=-1).numpy().tolist()
    labels, preds = [
        dataLoader.labelId2Tag(
            tf.argmax(tf.concat(ele, axis=0), axis=-1).numpy().tolist()
        )
        for ele in [labels, logits]
    ]
    sent_ids, sent_lens, range_ids = [
        tf.concat(ele, axis=0).numpy().tolist()
        for ele in [sent_ids, sent_lens, range_ids]
    ]
    samples = _filter_data(
        _collect_data(
            sent_ids,
            sent_lens,
            range_ids,
            probs,
            preds,
            labels,
        )
    )
    pred_seqs, true_seqs = [], []
    with open(f"./{dtype}_result.txt", "w") as fw:
        for sample in samples.values():
            length = sample["len"]
            pred_seq = iob_tagging(sample["p"], length)
            pred_seqs.extend(pred_seq)
            true_seq = iob_tagging(sample["t"], length)
            true_seqs.extend(true_seq)
            for t, p in zip(true_seq, pred_seq):
                fw.write(f"{t}\t{p}\n")
            fw.write("\n")
    return evaluate_ner(true_seqs, pred_seqs, verbose=verbose)


# Load the model weights
def load_model(path: str, models, dataLoader, optimizer, latent_dim):
    if os.path.exists(path):
        encoder, dec4txt, dec4img = models
        train_data = next(iter(dataLoader.Data("train")))
        train_one_batch4mvae_labeled(models, optimizer, train_data[:-3], latent_dim)
        encoder.load_weights(path, by_name="True")
