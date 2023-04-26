#!/bin/bash
set -x

export MULTIHOT_PREPROCESSED_DATASET='/home/vmagent/app/data/recsys2023_process/multihot'
export TOTAL_TRAINING_SAMPLES=3387880
export GLOBAL_BATCH_SIZE=256
export WORLD_SIZE=1



torchrun \
    --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=1 \
    --node_rank 0 \
    dlrm_main.py \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --synthetic_multi_hot_criteo_path $MULTIHOT_PREPROCESSED_DATASET \
    --num_embeddings_per_feature 137,6,634,7,5168,2,7,8,4,25,27,330,20,5802,11,50,902,20,56,35,25,5,5,4,3,3,3,3,3,3,5,2,2,2,2,2,2,2,2,2 \
    --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20))) \
    --epochs 1 \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --adagrad \
    --learning_rate 0.005