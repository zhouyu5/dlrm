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
from tqdm import tqdm
from sklearn import preprocessing


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


def get_df_from_filepath(data_dir, output_dir, preprocess=True, label_name='is_installed'):
    # input format
    # a. RowId(f_0)
    # b. Date(f_1)
    # c. Categorical features: 31 (f_2 to f_32)
    # d. Binary features: 9 (f_33 to f_41)
    # e. Numerical features: 38 (f_42 to f_79)
    # f. Labels(is_clicked, is_installed)

    # dense feature
    dense_feat_names = [f'f_{i}' for i in range(42, 80)]
    # dense cat feature
    dense_cat_feat_names = [f'cat_f_{i}' for i in range(42, 80)]
    # cat feature
    category_feat_names = [f'f_{i}' for i in range(2, 33)]
    # cat binary feat names
    cat_binary_feat_names = [f'f_{i}' for i in range(2, 42)]
    
    # read input
    all_files = glob.glob(data_dir)
    df = pd.concat((pd.read_csv(f, sep='\t') for f in all_files), ignore_index=True)

    # output format, dense: 38, sparse: 40
    save_cols = [label_name]
    save_cols += dense_feat_names + cat_binary_feat_names

    # process time: range, 45--67
    df['f_1'] = df['f_1'] - 45
    
    if preprocess:
        # process cat feature
        for feat in tqdm(category_feat_names, desc='Categorical Feature Processing'):
            df[feat] = df[feat].fillna(-1).astype('category').cat.codes
        
        # process dense feat
        # method1: min-max
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # method2: standard
        # scaler = preprocessing.StandardScaler()
        # method3: RobustScaler
        # scaler = preprocessing.RobustScaler()
        df[dense_feat_names] = pd.DataFrame(
            scaler.fit_transform(df[dense_feat_names]), 
            columns=dense_feat_names,
            index=df.index
        )

        # dense feature Discretize
        is_discretize = False
        if is_discretize:
            scaler = preprocessing.KBinsDiscretizer(
                n_bins=10, encode='ordinal', 
                strategy='uniform',
                random_state=2023
            )
            df[dense_cat_feat_names] = pd.DataFrame(
                scaler.fit_transform(df[dense_feat_names]), 
                columns=dense_cat_feat_names,
                index=df.index
            )

        df = df.fillna(0)

    for i in sorted(df['f_1'].unique()):
        print(f'processing day {i}')
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
    get_df_from_filepath(train_data_dir, output_dir, label_name='is_installed')

    # test_data_dir = f'{input_dir}/test/*.csv'
    # get_df_from_filepath(test_data_dir, output_dir)
    

if __name__ == "__main__":
    main(sys.argv[1:])
