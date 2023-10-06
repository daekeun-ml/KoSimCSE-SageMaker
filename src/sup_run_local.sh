#!/bin/bash

set -e
pip install -r requirements.txt

declare -a OPTS=(
    --base_model klue/roberta-base
    --dataset_dir ../dataset-sup-train
    --output_dir ../sup-model
    --batch_size 64
    --num_epochs 3
    --learning_rate 3e-5
    --temperature 0.05
    --lr_scheduler_type "linear"    
    --max_seq_len 64
    --eval_steps 50
    --seed 42
    --lora_dropout 0.05
)

SM_NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $SM_NUM_GPUS -eq 1 ]
then
    echo python train.py "${OPTS[@]}" "$@"
    CUDA_VISIBLE_DEVICES=0 python sup_train.py "${OPTS[@]}" "$@"
else
    echo torchrun --nnodes 1 --nproc_per_node "$SM_NUM_GPUS" sup_train_ddp.py "${OPTS[@]}" "$@"
    torchrun --nnodes 1 --nproc_per_node "$SM_NUM_GPUS" sup_train_ddp.py "${OPTS[@]}" "$@"
fi