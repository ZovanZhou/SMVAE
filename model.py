import tensorflow as tf
from typing import Tuple
from keras import initializers
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Dense, Bidirectional, LSTM, Lambda


class MVAEEncoder(tf.keras.models.Model):
    def __init__(self, bert_path: str, n_label: int, hidden_dim: int, latent_dim: int):
        super(MVAEEncoder, self).__init__()
        self.__n_label = n_label
        self.__ckpt_path = f"{bert_path}/bert_model.ckpt"
        self.__config_path = f"{bert_path}/bert_config.json"

        self.latent_dim = latent_dim
        self.bilstm1 = Bidirectional(LSTM(hidden_dim, return_sequences=True))
        self.bilstm2 = Bidirectional(LSTM(hidden_dim, return_sequences=True))
        self.fc1 = Dense(hidden_dim, activation=tf.nn.leaky_relu)
        self.fc2 = Dense(hidden_dim, activation=tf.nn.leaky_relu)
        self.fc3 = Dense(n_label, activation="softmax")
        self.fc4 = Dense(latent_dim)
        self.fc5 = Dense(latent_dim)
        self.fc6 = Dense(latent_dim)
        self.fc7 = Dense(latent_dim)
        self.fc8 = Dense(hidden_dim, activation=tf.nn.leaky_relu)
        # self.fc8 = Dense(hidden_dim * 8, activation=tf.nn.sigmoid)
        self.fc9 = Dense(hidden_dim)
        self.fc10 = Dense(hidden_dim)
        # self.fc11 = Dense(n_label, activation="softmax")

        self.bert_model = load_trained_model_from_checkpoint(
            self.__config_path, self.__ckpt_path, seq_len=None
        )
        for l in self.bert_model.layers:
            l.trainable = True

    @tf.function
    def extract_span_feature(self, contextual_feature, s_mask, s_len, e_mask, e_len):
        start_word_feature = tf.reduce_sum(contextual_feature * s_mask, axis=1) / s_len
        end_word_feature = tf.reduce_sum(contextual_feature * e_mask, axis=1) / e_len
        span_feature = tf.concat(
            [
                start_word_feature,
                end_word_feature,
                start_word_feature - end_word_feature,
                start_word_feature * end_word_feature,
            ],
            axis=-1,
        )
        return span_feature

    @tf.function
    def extract_span_global_feature(self, contextual_feature, full_mask):
        full_len = tf.reduce_sum(full_mask, axis=1)
        span_feature = tf.reduce_sum(contextual_feature * full_mask, axis=1) / full_len
        return span_feature

    def _product_of_experts(self, text_mu, text_logvar, img_mu, img_logvar):
        fusion_mu = (
            text_mu * tf.math.exp(img_logvar) + img_mu * tf.math.exp(text_logvar)
        ) / (tf.math.exp(img_logvar) + tf.math.exp(text_logvar))
        fusion_var = (
            tf.math.exp(text_logvar)
            * tf.math.exp(img_logvar)
            / (tf.math.exp(text_logvar) + tf.math.exp(img_logvar))
        )
        return (fusion_mu, tf.math.log(fusion_var))

    @tf.function
    def call(
        self, ind, seg, full_mask, s_mask, e_mask, s_len, e_len, img, training=True
    ):
        batch_size = img.get_shape().as_list()[0]
        full_mask, s_mask, e_mask, s_len, e_len = [
            tf.expand_dims(tf.cast(tensor, tf.float32), axis=-1)
            for tensor in [full_mask, s_mask, e_mask, s_len, e_len]
        ]
        # BERT features for samples
        original_contextual_feature = self.bert_model([ind, seg])
        original_span_feature = tf.stop_gradient(
            self.extract_span_global_feature(original_contextual_feature, full_mask)
            # self.extract_span_feature(
            #     original_contextual_feature, s_mask, s_len, e_mask, e_len
            # )
        )

        stop_gradient_contextual_feature = tf.stop_gradient(original_contextual_feature)
        contextual_feature1 = self.bilstm1(stop_gradient_contextual_feature)
        span_feature1 = self.extract_span_global_feature(contextual_feature1, full_mask)
        # self.extract_span_feature(
        #     contextual_feature1, s_mask, s_len, e_mask, e_len
        # )
        txt_hidden_feature = self.fc2(span_feature1)
        txt_mu = self.fc4(txt_hidden_feature)
        txt_logvar = self.fc5(txt_hidden_feature)

        original_img_feature = img
        img_hidden_feature = self.fc1(original_img_feature)
        img_mu = self.fc6(img_hidden_feature)
        img_logvar = self.fc7(img_hidden_feature)

        fusion_mu, fusion_logvar = self._product_of_experts(
            txt_mu, txt_logvar, img_mu, img_logvar
        )
        fusion_feature = fusion_mu + tf.math.exp(fusion_logvar / 2) * tf.random.normal(
            (batch_size, self.latent_dim)
        )

        contextual_feature2 = self.bilstm2(original_contextual_feature)
        span_feature2 = self.extract_span_feature(
            contextual_feature2, s_mask, s_len, e_mask, e_len
        )

        logits = self.fc3(self.fc8(tf.concat([fusion_feature, span_feature2], axis=-1)))

        return (
            original_span_feature,
            txt_mu,
            txt_logvar,
            original_img_feature,
            img_mu,
            img_logvar,
            fusion_mu,
            fusion_logvar,
            logits,
        )


class Decoder(tf.keras.models.Model):
    def __init__(self, hidden_dim: int, span_feature_dim: int):
        super(Decoder, self).__init__()
        self.fc1 = Dense(hidden_dim, activation=tf.nn.leaky_relu)
        self.fc2 = Dense(hidden_dim, activation=tf.nn.leaky_relu)
        self.fc3 = Dense(span_feature_dim)

    @tf.function
    def call(self, x):
        return self.fc3(self.fc2(self.fc1(x)))
