import os
import copy
from utils import getLogger
from typing import Dict, List


class DataParser(object):
    def __init__(
        self,
        path: str,
        n_selected_samples: int = -1,
    ) -> None:
        self.__base_path = path

        self.__dict_ner_label = []
        self.__logger = getLogger(self.__class__.__name__)

        _train_file_name = (
            "train.txt"
            if n_selected_samples <= 0
            else f"train_{n_selected_samples}_samples.txt"
        )
        _dev_file_name, _test_file_name = "dev.txt", "test.txt"

        raw_train_data_path, raw_dev_data_path, raw_test_data_path = [
            f"{self.__base_path}/{file_name}"
            for file_name in [_train_file_name, _dev_file_name, _test_file_name]
        ]

        gtd_train_data_path, gtd_dev_data_path, gtd_test_data_path = [
            f"{self.__base_path}/new_{file_name}"
            for file_name in [_train_file_name, _dev_file_name, _test_file_name]
        ]

        if n_selected_samples > 0:
            unlabeled_data_file_name = f"unlabeled_{n_selected_samples}_samples.txt"
            raw_unlabeled_data_path = f"{self.__base_path}/{unlabeled_data_file_name}"
            gtd_unlabeled_data_path = (
                f"{self.__base_path}/new_{unlabeled_data_file_name}"
            )
            if not os.path.exists(gtd_unlabeled_data_path):
                self.__logger.warning(
                    f"Could not find the parsed unlabeled data:{gtd_unlabeled_data_path}"
                )
                self.__logger.info(
                    f"Begin collecting raw unlabeled data from:{raw_unlabeled_data_path}"
                )
                unlabeled_raw_data = self.__get_sentences(raw_unlabeled_data_path)
                self.__generate_data(unlabeled_raw_data, gtd_unlabeled_data_path)
                self.__logger.info(
                    f"Finish, and save the parsed train data at:{gtd_unlabeled_data_path}"
                )

        if not os.path.exists(gtd_train_data_path):
            self.__logger.warning(
                f"Could not find the parsed train data:{gtd_train_data_path}"
            )
            self.__logger.info(
                f"Begin collecting raw train data from:{raw_train_data_path}"
            )
            train_raw_data = self.__get_sentences(raw_train_data_path)
            self.__generate_data(train_raw_data, gtd_train_data_path)
            self.__logger.info(
                f"Finish, and save the parsed train data at:{gtd_train_data_path}"
            )

        if not os.path.exists(gtd_dev_data_path):
            self.__logger.warning(
                f"Could not find the parsed development data:{gtd_dev_data_path}"
            )
            self.__logger.info(
                f"Begin collecting raw development data from:{raw_dev_data_path}"
            )
            dev_raw_data = self.__get_sentences(raw_dev_data_path)
            self.__generate_data(dev_raw_data, gtd_dev_data_path)
            self.__logger.info(
                f"Finish, and save the parsed development data at:{gtd_dev_data_path}"
            )

        if not os.path.exists(gtd_test_data_path):
            self.__logger.warning(
                f"Could not find the parsed test data:{gtd_test_data_path}"
            )
            self.__logger.info(
                f"Begin collecting raw test data from:{raw_test_data_path}"
            )
            test_raw_data = self.__get_sentences(raw_test_data_path)
            self.__generate_data(test_raw_data, gtd_test_data_path)
            self.__logger.info(
                f"Finish, and save the parsed test data at:{gtd_test_data_path}"
            )

    def __generate_data(self, raw_data: Dict, save_path: str) -> None:
        ner_tags = raw_data["ner"]
        imgids = raw_data["imgids"]
        sentences = raw_data["sentences"]
        cnt = 0

        with open(save_path, "w") as fw:
            for sentence, tag in zip(sentences, ner_tags):
                entity_tags = self.__get_entities(tag)
                for i in range(len(sentence)):
                    for j in range(i, len(sentence)):
                        tmp = [i, j]
                        label = tag[i].split("-")[-1] if tmp in entity_tags else "O"
                        fw.write(f"{imgids[cnt]}\n{sentence}\n{tmp}\n{label}\n\n")
                cnt += 1

    def __get_entities(self, tags: List) -> List:
        tmp = []
        entity_tags = []

        for i, t in enumerate(tags):
            if t != "O":
                if tmp:
                    if "B-" in t:
                        entity_tags.append(tmp.copy())
                        tmp.clear()
                        tmp.extend([i, i])
                    elif "I-" in t:
                        tmp[-1] = i
                else:
                    tmp.extend([i, i])
            else:
                if tmp:
                    entity_tags.append(tmp.copy())
                    tmp.clear()

        if tmp:
            entity_tags.append(tmp.copy())
        return entity_tags

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
                    elif "http" not in line:
                        tmp = line.split("\t")
                        word, r_tag = tmp[:2]
                        sentence.append(word)
                        ner_tag.append(r_tag)
                        if "train" in path and r_tag not in self.__dict_ner_label:
                            self.__dict_ner_label.append(r_tag)
                else:
                    sentences.append(copy.deepcopy(sentence))
                    ner_tags.append(copy.deepcopy(ner_tag))
                    sentence.clear()
                    ner_tag.clear()

        return {"sentences": sentences, "ner": ner_tags, "imgids": imgids}


if __name__ == "__main__":
    dataLoader = DataParser("./dataset/NCBI")
