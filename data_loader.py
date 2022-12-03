import os
import json
import codecs
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import getLogger
from keras_bert import Tokenizer
from tensorflow.data import Dataset
from typing import Dict, List, Tuple
from loader4image import ImageLoader
from keras.preprocessing.sequence import pad_sequences


class DataLoader(object):
    def __init__(
        self,
        data_path: str,
        bert_path: str,
        batch_size: int,
        n_selected_samples: int = -1,
    ) -> None:
        self.__logger = getLogger(self.__class__.__name__)
        self.__batch_size = batch_size
        self.__logger.info(f"Parameters [batch_size]:{self.__batch_size}")

        self.__dict_label = []

        self.__max_seq_len = 100
        self.__logger.info(f"Parameters [max_seq_len]:{self.__max_seq_len}")

        self.__base_path = data_path
        self.__logger.info(f"Dataset path:{self.__base_path}")
        label_path = f"{self.__base_path}/label.json"
        raw_train_data_path = (
            f"{self.__base_path}/new_train.txt"
            if n_selected_samples <= 0
            else f"{self.__base_path}/new_train_{n_selected_samples}_samples.txt"
        )
        ctd_train_data_path = (
            f"{self.__base_path}/new_train.json"
            if n_selected_samples <= 0
            else f"{self.__base_path}/new_train_{n_selected_samples}_samples.json"
        )
        raw_dev_data_path = f"{self.__base_path}/new_dev.txt"
        ctd_dev_data_path = f"{self.__base_path}/new_dev.json"
        raw_test_data_path = f"{self.__base_path}/new_test.txt"
        ctd_test_data_path = f"{self.__base_path}/new_test.json"

        if not os.path.exists(ctd_train_data_path):
            self.__logger.warning(
                f"Could not find train data file:{ctd_train_data_path}"
            )
            # Load the tokenizer for BERT
            self.__bert_path = bert_path
            self.__logger.info(
                f"Find the BERT path:{self.__bert_path}, and load the vocabulary"
            )
            self.__tokenizer = self.__load_vocabulary(f"{self.__bert_path}/vocab.txt")
            # Convert the original data to the tokenized tokens
            self.__logger.info(f"Converting the train set...")
            train_data = self.__get_data(raw_train_data_path)
            self.__convert_data(train_data, ctd_train_data_path)
            self.__logger.info(f"Finish, saving data path:{ctd_train_data_path}")
            # Save the label set
            self.__save_label(label_path)

        self.__load_label(label_path)

        if n_selected_samples > 0:
            raw_unlabeled_data_path = (
                f"{self.__base_path}/new_unlabeled_{n_selected_samples}_samples.txt"
            )
            ctd_unlabeled_data_path = (
                f"{self.__base_path}/new_unlabeled_{n_selected_samples}_samples.json"
            )
            if not os.path.exists(ctd_unlabeled_data_path):
                self.__logger.warning(
                    f"Could not find unlabeled data file:{ctd_unlabeled_data_path}"
                )
                # Load the tokenizer for BERT
                self.__bert_path = bert_path
                self.__logger.info(
                    f"Find the BERT path:{self.__bert_path}, and load the vocabulary"
                )
                self.__tokenizer = self.__load_vocabulary(
                    f"{self.__bert_path}/vocab.txt"
                )
                # Convert the original data to the tokenized tokens
                self.__logger.info(f"Converting the unlabeled set...")
                unlabeled_data = self.__get_data(raw_unlabeled_data_path)
                self.__convert_data(unlabeled_data, ctd_unlabeled_data_path)
                self.__logger.info(
                    f"Finish, saving data path:{ctd_unlabeled_data_path}"
                )

        if not os.path.exists(ctd_dev_data_path):
            self.__logger.warning(
                f"Could not find the development data file:{ctd_dev_data_path}"
            )
            # Convert the development set
            self.__logger.info(f"Converting the development set...")
            dev_data = self.__get_data(raw_dev_data_path)
            self.__convert_data(dev_data, ctd_dev_data_path)
            self.__logger.info(f"Finish, saving data path:{ctd_dev_data_path}")

        if not os.path.exists(ctd_test_data_path):
            self.__logger.warning(
                f"Could not find the test data file:{ctd_test_data_path}"
            )
            # Convert the test set
            self.__logger.info(f"Converting the test set...")
            test_data = self.__get_data(raw_test_data_path)
            self.__convert_data(test_data, ctd_test_data_path)
            self.__logger.info(f"Finish, saving data path:{ctd_test_data_path}")

        # Load the converted dataset
        if n_selected_samples > 0:
            self.__logger.info(
                f"Load the unlabeled data file:{ctd_unlabeled_data_path}"
            )
            self._unlabeled_data = self.__load_data(ctd_unlabeled_data_path)
        self.__logger.info(f"Load the train data file:{ctd_train_data_path}")
        self._train_data = self.__load_data(ctd_train_data_path)
        self.__logger.info(f"Load the development data file:{ctd_dev_data_path}")
        self._dev_data = self.__load_data(ctd_dev_data_path)
        self.__logger.info(f"Load the test data file:{ctd_test_data_path}")
        self._test_data = self.__load_data(ctd_test_data_path)

    @property
    def LABEL_SIZE(self) -> int:
        return len(self.__dict_label)

    @property
    def NONENTITY_LABEL_IDX(self) -> int:
        return self.__dict_label.index("O")

    @property
    def ENTITY_SIZE(self) -> int:
        return self.entity_size

    def labelId2Tag(self, idxs):
        return [self.__dict_label[i] for i in idxs]

    def resample_data(self, raw_data: Dict) -> Dict:
        data = raw_data.copy()
        label = data["label"]
        entity_idx = np.where(np.array(label) != self.NONENTITY_LABEL_IDX)[0]
        nonentity_idx = np.where(np.array(label) == self.NONENTITY_LABEL_IDX)[0]

        selected_idx = None
        if len(entity_idx) <= len(nonentity_idx):
            selected_nonentity_idx = np.random.choice(
                nonentity_idx, size=len(entity_idx), replace=False
            )
            selected_idx = np.concatenate((entity_idx, selected_nonentity_idx))
        else:
            selected_entity_idx = np.random.choice(
                entity_idx, size=len(nonentity_idx), replace=False
            )
            selected_idx = np.concatenate((selected_entity_idx, nonentity_idx))

        for k in data.keys():
            data[k] = np.array(data[k])[selected_idx]

        return data

    def Data(self, dtype: str) -> Dataset:
        return getattr(self, f"_{dtype}_data")

    def __save_label(self, save_path: str) -> None:
        with open(save_path, "w") as fw:
            json.dump(self.__dict_label, fw)

    def __load_label(self, path: str) -> None:
        with open(path, "r") as fr:
            self.__dict_label = json.load(fr)

    def __load_vocabulary(self, path: str) -> Tokenizer:
        token_dict = {}

        with codecs.open(path, "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        return Tokenizer(token_dict)

    def __tokenize_word(self, word: str, val: int) -> Tuple:
        ind, seg = self.__tokenizer.encode(first=word)
        CLS_idx, SEP_idx = ind[0], ind[-1]
        ind = ind[1:-1]
        seg = seg[1:-1]
        mask = [val] * len(ind)
        if len(ind) == 0:
            ind = [101]
            seg = [0]
            mask = [val] * len(ind)
        return (CLS_idx, SEP_idx, ind, seg, mask)

    def __convert_data(self, data: List, save_path: str, filter: bool = False) -> None:
        sample_indices = []
        sample_segments = []
        sample_start_mask = []
        sample_end_mask = []
        sample_full_mask = []
        sample_len_start_word = []
        sample_len_end_word = []
        sample_label = []
        sample_sentence_id = []
        sample_sentence_len = []
        sample_range_id = []
        dict_sentence = {}
        sample_imgids = []

        for sample in tqdm(data, ascii=True):
            imgid, sentence, idx, label = sample
            str_sentence = " ".join(sentence)
            if str_sentence not in dict_sentence:
                dict_sentence[str_sentence] = len(dict_sentence)

            entity_len = idx[1] - idx[0] + 1
            if label == "O" and (len(sentence) < 3 or entity_len > 5):
                continue

            sample_imgids.append(imgid)
            sample_sentence_id.append(dict_sentence[str_sentence])
            sample_sentence_len.append(len(sentence))
            sample_range_id.append([idx[0], idx[1]])

            CLS_idx, SEP_idx = 0, 0
            indices, segments = [], []
            start_mask, end_mask, full_mask = [], [], []
            len_start_word, len_end_word = 0, 0
            for i, w in enumerate(sentence):
                CLS_idx, SEP_idx, ind, seg, mask = self.__tokenize_word(
                    w, int(i in idx)
                )
                indices.extend(ind)
                segments.extend(seg)
                if i >= idx[0] and i <= idx[1]:
                    full_mask.extend([1] * len(mask))
                else:
                    full_mask.extend([0] * len(mask))
                if i == idx[0]:
                    start_mask.extend(mask)
                    len_start_word = len(ind)
                    end_mask.extend([1 - m for m in mask])
                elif i == idx[1]:
                    end_mask.extend(mask)
                    len_end_word = len(ind)
                    start_mask.extend([1 - m for m in mask])
                else:
                    start_mask.extend(mask)
                    end_mask.extend(mask)

            indices = [CLS_idx] + indices[: self.__max_seq_len - 2] + [SEP_idx]
            sample_indices.append(indices)

            if len_end_word == 0:
                end_mask = start_mask
                len_end_word = len_start_word

            segments, start_mask, end_mask = [
                [0] + ele[: self.__max_seq_len - 2] + [0]
                for ele in [segments, start_mask, end_mask]
            ]

            sample_segments.append(segments)
            sample_full_mask.append(full_mask)
            sample_start_mask.append(start_mask)
            sample_end_mask.append(end_mask)

            sample_len_start_word.append(len_start_word)
            sample_len_end_word.append(len_end_word)
            sample_label.append(self.__dict_label.index(label))

        (
            sample_indices,
            sample_segments,
            sample_full_mask,
            sample_start_mask,
            sample_end_mask,
        ) = [
            np.vstack(
                pad_sequences(ele, maxlen=self.__max_seq_len, padding="post", value=0)
            ).tolist()
            for ele in [
                sample_indices,
                sample_segments,
                sample_full_mask,
                sample_start_mask,
                sample_end_mask,
            ]
        ]

        with open(save_path, "w") as fw:
            json.dump(
                {
                    "indices": sample_indices,
                    "segments": sample_segments,
                    "full_mask": sample_full_mask,
                    "start_mask": sample_start_mask,
                    "end_mask": sample_end_mask,
                    "len_start_word": sample_len_start_word,
                    "len_end_word": sample_len_end_word,
                    "label": sample_label,
                    "sentence_id": sample_sentence_id,
                    "sentence_len": sample_sentence_len,
                    "range_id": sample_range_id,
                    "img_id": sample_imgids,
                },
                fw,
            )

    def reload_train_data(self) -> None:
        self._train_data = self.__convert_dict2dataset(
            self.resample_data(self._raw_train_data), True
        )

    def __convert_dict2dataset(
        self, raw_data: Dict, shuffle: bool = False, repeat: bool = False
    ) -> Dataset:
        # extract image features
        imgLoader = ImageLoader(f"{self.__base_path}/images")
        img_features = np.concatenate(
            [
                np.mean(imgLoader.getFeature(img_id), axis=1)
                for img_id in raw_data["img_id"]
            ],
            axis=0,
        )
        del raw_data["img_id"]

        raw_data["label"] = tf.one_hot(
            tf.constant(raw_data["label"], dtype=tf.int32),
            len(self.__dict_label),
        ).numpy()
        list_data = [np.array(data) for data in raw_data.values()]
        list_data.insert(7, img_features)
        dataset = Dataset.from_tensor_slices(tuple(list_data))
        if shuffle:
            dataset = dataset.shuffle(len(raw_data["label"]))
        if repeat:
            dataset = dataset.repeat()
        dataset = (
            dataset.prefetch(tf.data.experimental.AUTOTUNE)
            .batch(self.__batch_size, drop_remainder=shuffle)
            .cache()
        )
        return dataset

    def __load_data(self, path: str, flag: bool = True):
        with open(path, "r") as fr:
            raw_data = json.load(fr)
            shuffle_flag = False
            repeat_flag = False
            if "unlabeled" in path:
                shuffle_flag = True
                repeat_flag = True
            if "train" in path:
                shuffle_flag = True
                self._raw_train_data = raw_data
                self.entity_size = len(
                    np.where(
                        np.array(raw_data["label"]) != self.__dict_label.index("O")
                    )[0]
                )
                raw_data = self.resample_data(raw_data)
            return (
                self.__convert_dict2dataset(raw_data, shuffle_flag, repeat_flag)
                if flag
                else raw_data
            )

    def __get_data(self, path: str) -> List:
        cnt = 0
        data = []

        with open(path, "r") as fr:
            sample = []
            for line in fr.readlines():
                line = line.strip("\n")
                if line:
                    if cnt == 0:
                        sample.append(line.split(":")[-1])
                    elif cnt == 3:
                        sample.append(line)
                    else:
                        sample.append(eval(line))
                    cnt = (cnt + 1) % 4
                else:
                    if "train" in path and sample[-1] not in self.__dict_label:
                        self.__dict_label.append(sample[-1])
                    if sample[-1] in self.__dict_label:
                        data.append(tuple(sample))
                    sample.clear()

        return data


if __name__ == "__main__":
    dataLoader = DataLoader(
        data_path="./dataset/NCBI",
        bert_path="../E2EMERN/biobert_large",
        batch_size=10,
    )
    print(len(dataLoader.Data("train")))
