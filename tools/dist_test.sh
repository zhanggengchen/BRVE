#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}

# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=1 --master_port=29500 tools/test.py configs/BRVE_LLRVD.py pretrained_models/BRVE_LLRVD_new.pth --seed 0 --save-path test_results/BRVE_LLRVD --out test_results/BRVE_LLRVD/result.json --launcher pytorch
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=1 --master_port=29500 tools/test.py configs/BRVE_SMOID.py pretrained_models/BRVE_SMOID_new.pth --seed 0 --save-path test_results/BRVE_SMOID --out test_results/BRVE_SMOID/result.json --launcher pytorch