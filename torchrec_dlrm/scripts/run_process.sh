#!/bin/bash
set -x

npy_data_dir='/home/vmagent/app/data/recsys2023_process/numpy_contiguous_shuffled_output_dataset_dir'
multihot_data_dir='/home/vmagent/app/data/recsys2023_process/multihot'


bash scripts/process_recsys.sh \
   /home/vmagent/app/data/recsys2023_process/raw \
   /home/vmagent/app/data/recsys2023_process/temp \
   $npy_data_dir



python scripts/materialize_synthetic_multihot_dataset.py \
    --in_memory_binary_criteo_path $npy_data_dir \
    --output_path $multihot_data_dir \
    --num_embeddings_per_feature 137,6,634,7,5168,2,7,8,4,25,27,330,20,5802,11,50,902,20,56,35,25,5,5,4,3,3,3,3,3,3,5,2,2,2,2,2,2,2,2,2 \
    --multi_hot_sizes 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 \
    --multi_hot_distribution_type uniform \
    --copy_labels_and_dense