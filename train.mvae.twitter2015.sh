CUDA_VISIBLE_DEVICES=0 \
python train.mvae.py \
    --dataset twitter2015 \
    --lr 1e-5 \
    --patience 10 \
    --gamma 0.025 \
    --n_selected_samples 100 \
    --latent_dim 100 \
    --seed 0 \
    --save_model 1 \
    --semi_supervised 1 \
    --weights ./weights/twitter2015.model.h5 \
    --mode test \
