#!/bin/bash
set -x

# bash scripts/run_process.sh && bash scripts/run_train.sh

export CUDA_VISIBLE_DEVICES=1
# export in_memory_binary_criteo_path='/home/vmagent/app/data/recsys2023_process/numpy_contiguous_shuffled_output_dataset_dir'
# export in_memory_binary_criteo_path='/home/vmagent/app/data/recsys2023_process/npy-base'
export in_memory_binary_criteo_path='/home/vmagent/app/data/recsys2023_process/npy-new-base-leaf-50'
export WORLD_SIZE=1
# export GLOBAL_BATCH_SIZE=256
export GLOBAL_BATCH_SIZE=128
learning_rate=0.01
epochs=1



# --interaction_type=dcn \
# --dcn_num_layers=3 \
# --dcn_low_rank_dim=128 \

torchrun \
    --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=1 \
    dlrm_main.py \
    --embedding_dim 16 \
    --dense_arch_layer_sizes 256,128,64,32,16 \
    --over_arch_layer_sizes 512,256,1 \
    --in_memory_binary_criteo_path $in_memory_binary_criteo_path \
    --epochs $epochs \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --adagrad \
    --learning_rate $learning_rate \
    --allow_tf32 \
    --tasks "train,val" \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=128 \
    --loss_type "BCE" \
    --shuffle_batches
