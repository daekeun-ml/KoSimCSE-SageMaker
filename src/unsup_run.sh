#!/bin/bash

set -e
pip install -r requirements.txt

declare -a OPTS=(
    --base_model klue/roberta-base
    #--dataset_dir /opt/ml/input/data/training/
    --output_dir /opt/ml/model/
    --batch_size 64
    --num_epochs 1
    --learning_rate 1e-5
    --temperature 0.05
    --lr_scheduler_type "linear"    
    --max_seq_len 32
    --eval_steps 50
    --seed 42
    --lora_dropout 0.05
)

if [ $SM_NUM_GPUS -eq 1 ]
then
    echo python unsup_train.py "${OPTS[@]}" "$@"
    CUDA_VISIBLE_DEVICES=0 python unsup_train.py "${OPTS[@]}" "$@"
else
    echo torchrun --nnodes 1 --nproc_per_node "$SM_NUM_GPUS" unsup_train_ddp.py "${OPTS[@]}" "$@"
    torchrun --nnodes 1 --nproc_per_node "$SM_NUM_GPUS" unsup_train_ddp.py "${OPTS[@]}" "$@"
fi