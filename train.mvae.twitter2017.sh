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
    --mode test \