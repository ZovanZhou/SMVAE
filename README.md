# SMVAE

[Title] A Span-based Multimodal Variational Autoencoder for Semi-supervised Multimodal Named Entity Recognition

[Authors] Baohang Zhou, Ying Zhang, Kehui Song, Xuhui Sui, Guoqing Zhao, Hongbin Wang and Xiaojie Yuan

[EMNLP 2022 Main Conference]

## Preparation

1. Clone the repo to your local.
2. Download Python version: 3.6.13
3. Download the dataset from this [link](https://pan.baidu.com/s/1mM1U82XOt663AGnkk0QzmQ) and the extraction code is **1234**. Put the downloaded files into the ''**dataset**'' folder.
4. Download the [BERT](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip) pretrained models, and unzip into the ''**pretrain**'' folder.
5. Open the shell or cmd in this repo folder. Run this command to install necessary packages.

```cmd
pip install -r requirements.txt
```

## Experiments

1. For Linux systems, we have shell scripts to run the training procedures. You can run the following command:

```cmd
./train.mvae.twitter2015.sh

or

./train.mvae.twitter2017.sh
```

2. You can also input the following command to train the model. There are different choices for some hyper-parameters shown in square barckets. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
| epoch | int | Training times |
| patience | int | Early stopping |
| gamma | float | The percentage of |
| save_model | int | Whether to save the training model |
| latent_dim | int | Dimension number of latent variable |
| semi_supervised | int | Whether to run semi-supervised model |
| n_selected_samples | int | Number of labeled data samples |
| weights | string | Saved model path |
| dataset | string | Dataset name |
| mode | string | To train or test model |

```cmd
CUDA_VISIBLE_DEVICES=1 \
python train.mvae.py \
    --dataset twitter2017 \
    --seed 6 \
    --lr 1e-5 \
    --epoch 100 \
    --gamma 0.03 \
    --patience 20 \
    --save_model 1 \
    --latent_dim 100 \
    --semi_supervised 1 \
    --n_selected_samples 100 \
    --weights ./weights/twitter2017.model.h5 \
    --mode train \
```

> When ''**n_selected_samples**'' parameter is set to -1, the all samples in ''train.txt'' will be used to train the model.

3. After training the model, you can change the ''**mode**'' parameter to ''test'' for evaluating the model on the test set.

4. We also provide the weights of the model to reimplement the results in our paper. You should download the weight files from this [link](https://pan.baidu.com/s/1mM1U82XOt663AGnkk0QzmQ) and the extraction code is **1234**, then put them into the ''**weights**'' folder. You can run the script directly after downloading the dataset and weights.

5. When you change the ''**n_selected_samples**'' parameter to run different experiments, you should delete the files except for ''**train.txt, dev.txt, valid.txt, test.txt, images/**'' in the ''**./dataset/twitter2015/**'' or ''**./dataset/twitter2017/**'' folders.
