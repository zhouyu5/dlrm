#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script preprocesses Criteo dataset tsv files to binary (npy) files.

import argparse
import os
import sys
from typing import List
import pandas as pd
import glob
import tqdm


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Criteo tsv -> npy preprocessing script."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing Criteo tsv files."
        "For criteo_1tb, files in the directory should be named day_{0-23}."
        "For criteo_kaggle, files in the directory should be train.txt & test.txt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to store npy files.",
    )
    return parser.parse_args(argv)


def get_df_from_filepath(data_dir, output_dir, preprocess=False, label_name='is_installed', is_test=False):
    # a. RowId(f_0)
    # b. Date(f_1)
    # c. Categorical features: 31 (f_2 to f_32)
    # d. Binary features: 9 (f_33 to f_41)
    # e. Numerical features: 38 (f_42 to f_79)
    # f. Labels(is_clicked, is_installed)
    all_files = glob.glob(data_dir)

    if label_name == 'is_installed':
        drop_label_name = 'is_clicked'
    else:
        drop_label_name = 'is_installed'

    # dense: 38, sparse: 40
    save_cols = [f'f_{i}' for i in range(42, 80)] + [f'f_{i}' for i in range(2, 42)]
    if not is_test:
        save_cols = [label_name] + save_cols
    else:
        save_cols = ['f_0'] + save_cols

    df = pd.concat((pd.read_csv(f, sep='\t') for f in all_files), ignore_index=True)

    if not is_test:
        df = df.drop(columns=[drop_label_name])

    df['f_1'] = df['f_1'] - 45
    
    if preprocess:
        category_feat_names = [f'f_{i}' for i in range(2,33)]
        for feat in tqdm(category_feat_names, desc='Categorical Feature Processing'):
            df[feat] = df[feat].astype('category').cat.codes

    for i in df['f_1'].unique():
        df_temp = df[df['f_1'] == i]
        df_temp = df_temp[save_cols]
        save_path = f'{output_dir}/day_{i}'
        df_temp.to_csv(save_path, sep='\t', header=False, index=False)
    
    return


def main(argv: List[str]) -> None:
    """
    This function preprocesses the raw Criteo tsvs into the format (npy binary)
    expected by InMemoryBinaryCriteoIterDataPipe.

    Args:
        argv (List[str]): Command line args.

    Returns:
        None.
    """
    args = parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.system(f'mkdir -p {output_dir}')

    train_data_dir = f'{input_dir}/train/*.csv'
    get_df_from_filepath(train_data_dir, output_dir, label_name='is_installed', is_test=False)

    test_data_dir = f'{input_dir}/test/*.csv'
    get_df_from_filepath(test_data_dir, output_dir, is_test=True)
    

if __name__ == "__main__":
    main(sys.argv[1:])


# python data/combine_recsys.py --input_dir '/home/vmagent/app/data/sharechat_recsys2023_data' --output_dir '/home/vmagent/app/data/recsys2023_process/raw'
    
    

