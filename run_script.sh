#!/bin/bash
# run_script.sh

DATASET_NAME="TINY_IMAGENET"
EXP_NAME="runs/tiny_imagenet_clip_finetune"
SRC_TXT="a photo"
TRG_TXT="a beautiful landscape"
N_ITER=1
T_0=500            # return step in [0, 1000): how much noise the edit starts from
N_INV_STEP=40      # DDIM steps used to invert images into latents
N_TRAIN_STEP=6     # DDIM steps of the generative process during fine-tuning
N_TEST_STEP=40     # DDIM steps of the generative process at evaluation
N_TRAIN_IMAGE=1
LOSS_W_CLIP=3.0
LOSS_W_L1=0.0
LAMBDA_DIRECTION=1.0  # mean directional CLIP loss
LAMBDA_OT=0.0         # optimal-transport (Sinkhorn) directional CLIP loss, e.g. 1.0

echo "Starting DiffusionCLIP Training..."
echo "Dataset: $DATASET_NAME"
echo "Target Text: $TRG_TXT"

# Provide parameters for the underlying script
python main.py --config configs/tiny_imagenet.yml \
    --clip_finetune_eff \
    --exp "$EXP_NAME" \
    --src_txts "$SRC_TXT" \
    --trg_txts "$TRG_TXT" \
    --do_train 1 \
    --do_test 1 \
    --n_iter $N_ITER \
    --t_0 $T_0 \
    --n_inv_step $N_INV_STEP \
    --n_train_step $N_TRAIN_STEP \
    --n_test_step $N_TEST_STEP \
    --n_precomp_img $N_TRAIN_IMAGE \
    --n_train_img $N_TRAIN_IMAGE \
    --n_test_img 1 \
    --clip_loss_w $LOSS_W_CLIP \
    --l1_loss_w $LOSS_W_L1 \
    --id_loss_w 0.0 \
    --lambda_direction $LAMBDA_DIRECTION \
    --lambda_ot $LAMBDA_OT \
    --lr_clip_finetune 2e-6 \
    --save_train_image 1

echo "Training Completed."
