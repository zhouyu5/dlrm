#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=1
export MULTIHOT_PREPROCESSED_DATASET='/home/vmagent/app/data/recsys2023_process/multihot'
export GLOBAL_BATCH_SIZE=128
export WORLD_SIZE=1
num_embeddings_per_feature='136,5,633,6,5167,1,6,7,3,24,26,329,19,5801,10,49,901,19,55,34,24,4,4,3,2,2,2,2,3,3,4,2,2,2,2,2,2,2,2,2'
learning_rate=0.01
epochs=10

torchrun \
    --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=1 \
    dlrm_main.py \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,512,256,1 \
    --synthetic_multi_hot_criteo_path $MULTIHOT_PREPROCESSED_DATASET \
    --num_embeddings_per_feature $num_embeddings_per_feature \
    --epochs $epochs \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --adagrad \
    --learning_rate $learning_rate