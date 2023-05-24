#!/bin/bash
set -x

raw_data_dir='/home/vmagent/app/data/recsys2023_process/raw'
npy_data_dir='/home/vmagent/app/data/recsys2023_process/numpy_contiguous_shuffled_output_dataset_dir'
temp_data_dir='/home/vmagent/app/data/recsys2023_process/temp'
multihot_data_dir='/home/vmagent/app/data/recsys2023_process/multihot'

# step 1: raw data
python data/combine_recsys.py \
   --input_dir '/home/vmagent/app/data/sharechat_recsys2023_data' \
   --output_dir $raw_data_dir

# python data/combine_recsys.py \
#    --input_dir '/home/vmagent/app/data/sharechat_recsys2023_data' \
#    --output_dir '/home/vmagent/app/data/recsys2023_process/raw8'


# step 2: one-hot data
bash scripts/process_recsys.sh \
   $raw_data_dir \
   $temp_data_dir \
   $npy_data_dir

# # step 3: multi-hot data
# python data/materialize_synthetic_multihot_dataset.py \
#     --in_memory_binary_criteo_path $npy_data_dir \
#     --output_path $multihot_data_dir \
#     --copy_labels_and_dense