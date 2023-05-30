#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script preprocesses Criteo dataset tsv files to binary (npy) files.

import pandas as pd
import numpy as np
import glob
import math
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from category_encoders import *
import os
import sys
import argparse
from typing import List


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


def get_train_test_df(input_dir, test_date):
    def get_df_from_path(path):
        df = pd.read_parquet(path)
        gp_columns = ['f2_4', 'f4_6', 'f13_18', 'f2_19_to_22', 
                      'f23_to_29', 'f43_51_58_59_64_to_70', 
                      'f44_to_47', 'f48_to_50', 'f71_33', 'f72_34', 
                      'f73_35', 'f74_36', 'f75_37', 'f76_38', 
                      'f77_to_79']
        # gp_after_columns = list(map(lambda x: f'gp_{x}', gp_columns))
        # df[gp_after_columns] = df[gp_columns]

        drop_columns = gp_columns + ['f2_4_is_installed_GCE', 'f_7_is_installed_CE']
            # [column for column in df.columns if column.endswith('CE')]
        df = df.drop(drop_columns, axis=1)
        return df
        
    train_data_path = f"{input_dir}/before_day_{test_date}_is_installed_FE_2.parquet"    
    test_data_path = f"{input_dir}/day_{test_date}_is_installed_FE_2.parquet"    

    train_df = get_df_from_path(train_data_path)
    test_df = get_df_from_path(test_data_path)
    if 'is_installed' not in test_df.columns:
        test_df['is_clicked'] = test_df['is_installed'] = test_df['f_0']

    print('before process shape')
    print(train_df.shape, test_df.shape)
    return train_df, test_df


def categorify_cat_feat(df_train, df_test=None):
    def get_cat_id(cat_columns, df_train, is_combine=True, df_test=None):
        if not cat_columns:
            return df_train, df_test
        cat_enc = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            (
                "categorify-1", 
                 preprocessing.OrdinalEncoder(
                     dtype=np.int64,
                     # encoded_missing_value=-1,
                     handle_unknown='use_encoded_value',
                     unknown_value=-1,
                     # min_frequency=2,
                 )
            ),
        ])

        if is_combine and df_test is not None:
            df_all = pd.concat((df_train, df_test), ignore_index=True)
            cat_enc.fit(df_all[cat_columns])
            del df_all
        else:
            cat_enc.fit(df_train[cat_columns])

        df_train[cat_columns] = cat_enc.transform(df_train[cat_columns])
        if df_test is not None:
            df_test[cat_columns] = cat_enc.transform(df_test[cat_columns])
        
        if not is_combine:
            for column in cat_columns:
                if -1 in df_test[column].unique():
                    df_train[column] += 1
                    df_test[column] += 1
        return df_train, df_test
        
    basic_cat_columns = [f'f_{i}' for i in range(2, 33)]
    group_columns = [
        column for column in df_train.columns
        if column.startswith('gp_cnt')
    ]
    
    df_train, df_test = get_cat_id(basic_cat_columns, df_train, is_combine=IS_COMBINE, df_test=df_test)
    df_train, df_test = get_cat_id(group_columns, df_train, is_combine=IS_COMBINE, df_test=df_test)
    
    return df_train, df_test


def missing_dense_feat(df_train, df_test=None):
    dense_columns = [f'f_{i}' for i in range(42, 80)]
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[dense_columns] = df_train[dense_columns].fillna(0)
    if df_test is not None:
        df_test[dense_columns] = df_test[dense_columns].fillna(0)
        
    return df_train, df_test


def process_cat_feat(df_train, df_test=None):
    
    df_train, df_test = categorify_cat_feat(df_train, df_test)
            
    return df_train, df_test


def min_max_dense_feat(df_train, df_test=None):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    dense_columns = [f'f_{i}' for i in range(33, 80)] + \
        [column for column in df_train.columns if column.endswith('CE')]

    df_train[dense_columns] = scaler.fit_transform(df_train[dense_columns])
    if df_test is not None:
        df_test[dense_columns] = scaler.transform(df_test[dense_columns])

    return df_train, df_test
    

def process_dense_feat(df_train, df_test=None):  
    df_train, df_test = min_max_dense_feat(df_train, df_test)  
    return df_train, df_test
    
    
def get_processed_df(df_train, df_test=None):

    # step1: process dense missing value
    df_train, df_test = missing_dense_feat(df_train, df_test)
    
    # step2: process cat feat
    print('processing cat feature')
    df_train, df_test = process_cat_feat(df_train, df_test)

    # step3: process numerical feat
    print('processing dense feature')
    df_train, df_test = process_dense_feat(df_train, df_test)
    
    return df_train, df_test


def save_output_df(df_train, df_test, test_date, output_dir):

    if test_date == 67:
        valid_date = 66
        df_valid = df_train.loc[df_train['f_1'] == valid_date]
    else:
        valid_date = test_date
        df_valid = df_test

    label_cols = [
        'is_installed',
        'is_clicked',
        'f_0',
        'f_1',
    ]

    dense_columns = [f'f_{i}' for i in range(33, 80)] + \
        [column for column in df_train.columns if column.endswith('CE')]
    
    cat_columns = [f'f_{i}' for i in range(2, 7)] + \
        [f'f_{i}' for i in range(8, 33)] + \
        [column for column in df_train.columns if column.startswith('gp_')]

    save_cols = label_cols + dense_columns + cat_columns

    print(f'dense columns: {len(dense_columns)}, cat columns: {len(cat_columns)}, total_columns: {save_cols}')

    train_save_path = f'{output_dir}/train'
    valid_save_path = f'{output_dir}/valid'
    test_save_path = f'{output_dir}/test'
    df_train[save_cols].to_csv(train_save_path, sep='\t', header=False, index=False)
    df_valid[save_cols].to_csv(valid_save_path, sep='\t', header=False, index=False)
    df_test[save_cols].to_csv(test_save_path, sep='\t', header=False, index=False)


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

    train_df, test_df = get_train_test_df(input_dir, TEST_DATE)

    train_df, test_df = get_processed_df(train_df, test_df)

    save_output_df(train_df, test_df, TEST_DATE, output_dir)

    return
    

if __name__ == "__main__":
    TEST_DATE = 60
    IS_COMBINE = True
    main(sys.argv[1:])


# python data/combine_recsys_3.py \
#    --input_dir '/home/vmagent/app/data/LGBM_FE2' \
#    --output_dir '/home/vmagent/app/data/recsys2023_process/raw11'