import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, Tuple
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input


class ImageLoader(object):
    def __init__(self, path: str) -> None:
        self.__feature_path = f"{path}/dict_feature.pkl"

        if not os.path.exists(self.__feature_path):
            imgs = self.__read_data(path)
            features = self.__parse_data(imgs, self.__feature_path)
        else:
            features = pd.read_pickle(self.__feature_path)
        self.__dict_features = features
        self.__feature_shape = list(self.__dict_features.values())[0].shape

    def getFeature(self, imgid: str):
        return self.__dict_features.get(imgid, None).reshape(
            (1, -1, self.__feature_shape[-1])
        )

    def __parse_data(self, data: Tuple, save_path: str) -> Dict:
        imgids, imgs = data
        tf_data = (
            Dataset.from_tensor_slices((imgs,))
            .batch(64)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .cache()
        )
        model = ResNet152(weights="imagenet", include_top=False)
        features = []
        for batch_data in tqdm(tf_data, ncols=80, ascii=True):
            feature = model.predict(batch_data)
            features.append(feature)
        features = np.vstack(features)
        dict_features = {}
        for imgid, feature in zip(imgids, features):
            dict_features[imgid] = feature
        pd.to_pickle(dict_features, save_path)
        return dict_features

    def __read_data(self, path: str):
        file_names = os.listdir(path)
        imgids, imgs = [], []
        for file_name in file_names:
            try:
                img = image.load_img(f"{path}/{file_name}", target_size=(224, 224))
            except Exception as ex:
                img = image.load_img(f"{path}/17_06_4705.jpg", target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            imgs.append(x)
            imgids.append(file_name.split(".")[0])
        imgs = np.vstack(imgs)
        return (imgids, imgs)


if __name__ == "__main__":
    imgLoader = ImageLoader("./dataset/twitter2015/images")