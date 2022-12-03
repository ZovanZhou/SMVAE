import os
import copy
import numpy as np
from utils import getLogger
from typing import Dict, List


class DataSelector(object):
    def __init__(self, path: str, n_selected_samples: int = 100):
        self.__base_path = path
        self.__n_selected_samples = n_selected_samples
        self.__logger = getLogger(self.__class__.__name__)

        raw_train_data_path = f"{self.__base_path}/train.txt"
        selected_train_data_path = (
            f"{self.__base_path}/train_{n_selected_samples}_samples.txt"
        )
        selected_unlabeled_data_path = (
            f"{self.__base_path}/unlabeled_{n_selected_samples}_samples.txt"
        )

        if not (
            os.path.exists(selected_train_data_path)
            and os.path.exists(selected_unlabeled_data_path)
        ):
            self.__logger.warning(
                f"Could not find the selected train data:{selected_train_data_path}"
            )
            self.__logger.info(f"Begin selecting train data from:{raw_train_data_path}")
            self.__select_samples(
                self.__get_sentences(raw_train_data_path),
                selected_train_data_path,
                selected_unlabeled_data_path,
            )
            self.__logger.info(
                f"Finish, and save the selected train data at:{selected_train_data_path}"
            )

    def __select_samples(
        self, data: Dict, train_data_path: str, unlabeled_data_path: str
    ) -> None:
        sentences = data["sentences"]
        imgids = data["imgids"]
        ner_tags = data["ner"]
        samples_idx = [i for i in range(len(sentences))]

        selected_idx = np.random.choice(
            samples_idx, self.__n_selected_samples, replace=False
        )

        with open(train_data_path, "w") as fw1, open(unlabeled_data_path, "w") as fw2:
            for i in range(len(samples_idx)):
                fw = fw1 if i in selected_idx else fw2
                imgid = imgids[i]
                fw.write(f"{imgid}\n")
                for word, ner_tag in zip(sentences[i], ner_tags[i]):
                    fw.write(f"{word}\t{ner_tag}\n")
                fw.write("\n")

    def __get_sentences(self, path: str) -> Dict:
        """
        handle the input data files to the list form
        """
        imgids = []
        ner_tags = []
        sentences = []

        with open(path, "r") as fp:
            ner_tag = []
            sentence = []

            for line in fp.readlines():
                line = line.strip()

                if line:
                    if "IMGID" in line:
                        imgids.append(line)
                    else:
                        word, r_tag = line.split("\t")
                        sentence.append(word)
                        ner_tag.append(r_tag)
                else:
                    sentences.append(copy.deepcopy(sentence))
                    ner_tags.append(copy.deepcopy(ner_tag))
                    sentence.clear()
                    ner_tag.clear()
        assert len(imgids) == len(sentences)
        return {
            "sentences": sentences,
            "imgids": imgids,
            "ner": ner_tags,
        }
