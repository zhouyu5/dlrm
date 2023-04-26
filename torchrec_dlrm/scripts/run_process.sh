#!/bin/bash
set -x

raw_data_dir='/home/vmagent/app/data/recsys2023_process/raw'
npy_data_dir='/home/vmagent/app/data/recsys2023_process/numpy_contiguous_shuffled_output_dataset_dir'
temp_data_dir='/home/vmagent/app/data/recsys2023_process/temp'
multihot_data_dir='/home/vmagent/app/data/recsys2023_process/multihot'
num_embeddings_per_feature='136,5,633,6,5167,1,6,7,3,24,26,329,19,5801,10,49,901,19,55,34,24,4,4,3,2,2,2,2,3,3,4,2,2,2,2,2,2,2,2,2'


python data/combine_recsys.py \
   --input_dir '/home/vmagent/app/data/sharechat_recsys2023_data' \
   --output_dir $raw_data_dir


bash scripts/process_recsys.sh \
   $raw_data_dir \
   $temp_data_dir \
   $npy_data_dir


python scripts/materialize_synthetic_multihot_dataset.py \
    --in_memory_binary_criteo_path $npy_data_dir \
    --output_path $multihot_data_dir \
    --num_embeddings_per_feature $num_embeddings_per_feature \
    --multi_hot_sizes '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1' \
    --multi_hot_distribution_type uniform \
    --copy_labels_and_dense