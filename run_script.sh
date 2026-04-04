#!/bin/bash
# run_script.sh

DATASET_NAME="TINY_IMAGENET"
# Determine which class to target if relevant (e.g. 0-199 for tiny imagenet) or just use general diffusion
EXP_NAME="tiny_imagenet_clip_finetune"
SRC_TXT="a photo"
TRG_TXT="a beautiful landscape"
N_ITER=1
STEPS=20
N_TRAIN_IMAGE=1
LOSS_W_CLIP=3.0
LOSS_W_L1=0.0

echo "Starting DiffusionCLIP Training..."
echo "Dataset: $DATASET_NAME"
echo "Target Text: $TRG_TXT"

# Provide parameters for the underlying script
python main.py --config configs/tiny_imagenet.yml \
    --exp $EXP_NAME \
    --edit_attr "" \
    --src_txts "$SRC_TXT" \
    --trg_txts "$TRG_TXT" \
    --do_train 1 \
    --do_test 1 \
    --n_iter $N_ITER \
    --t_0 $STEPS \
    --n_inv_step $STEPS \
    --n_train_step $STEPS \
    --n_test_step $STEPS \
    --n_train_img $N_TRAIN_IMAGE \
    --n_test_img 1 \
    --clip_loss_w $LOSS_W_CLIP \
    --l1_loss_w $LOSS_W_L1 \
    --id_loss_w 0.0 \
    --lr_clip_finetune 2e-6 \
    --save_train_image

echo "Training Completed."
