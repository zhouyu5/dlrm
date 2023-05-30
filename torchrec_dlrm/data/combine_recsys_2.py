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


def get_df_from_filepath(data_path):
    all_files = glob.glob(data_path)
    df = pd.concat((pd.read_csv(f, sep='\t') for f in all_files), ignore_index=True)        
    return df


def get_train_test_df(input_dir, test_date):
    train_data_path = f'{input_dir}/train/*.csv'    
    test_data_path = f'{input_dir}/test/*.csv'    

    train_df = get_df_from_filepath(train_data_path)
    test_df = get_df_from_filepath(test_data_path)
    test_df['is_clicked'] = test_df['is_installed'] = test_df['f_0']
    total_df = pd.concat((train_df, test_df), ignore_index=True)

    test_df = total_df.loc[total_df['f_1'] == test_date]
    train_df = total_df.loc[total_df['f_1'] < test_date]

    print('before process shape')
    print(train_df.shape, test_df.shape)
    return train_df, test_df


def get_group_dict():
    group_dict = {
        'cnt': [
            ['f_4', 'f_6'],
            ['f_13', 'f_18'],
            ['f_2', 'f_19', 'f_20', 'f_21', 'f_22'],
            ['f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29'],
            # ['f_43', 'f_51', 'f_58', 'f_59', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69','f_70'],
            # ['f_44', 'f_45', 'f_46','f_47'],
            # ['f_48', 'f_49', 'f_50'],   
            # ['f_71', 'f_33'],
            # ['f_72', 'f_34'],
            # ['f_73', 'f_35'],
            # ['f_74', 'f_36'],
            # ['f_75', 'f_37'],
            # ['f_76', 'f_38'],
            # ['f_77', 'f_39', 'f_78', 'f_40', 'f_79', 'f_41'],
        ],
        'click': [
            ['f_2'], 
            ['f_4', 'f_6'],
            ['f_2', 'f_19', 'f_20', 'f_21', 'f_22'],
        ],
    }
    
    return group_dict


def group_cat_feat(df_train, df_test=None):
    def get_group_column(df, group):
        return df.apply(
            lambda x: ','.join(map(str, [float(x[key]) for key in group])),
            axis=1
        )
    
    group_dict = get_group_dict()
    for key, group_list in group_dict.items():
        for i, group in enumerate(group_list):
            group_name = f'gp_{key}_{i}'
            df_train[group_name] = get_group_column(df_train, group)
            if df_test is not None:
                df_test[group_name] = get_group_column(df_test, group)
            
    return df_train, df_test


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


def onehot_cat_feat(df_train, df_test=None):
    num_per_cat_dict_train = {}
    basic_cat_columns = [f'f_{i}' for i in range(2, 33)]
    for feat_name in basic_cat_columns:
        num_per_cat_dict_train[feat_name] = df_train[feat_name].nunique()
     
    onehot_before_columns = sorted([
        key for key, value in num_per_cat_dict_train.items() 
        if 3 <= value < 10
    ])

    onehot_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    onehot_enc.fit(df_train[onehot_before_columns])
    onehot_after_columns = onehot_enc.get_feature_names_out(onehot_before_columns)
    onehot_after_columns = list(map(lambda x: f'onehot_{x}', onehot_after_columns))
    # print(onehot_after_columns)
    
    df_train[onehot_after_columns] = onehot_enc\
        .transform(df_train[onehot_before_columns])\
        .toarray()
    if df_test is not None:
        df_test[onehot_after_columns] = onehot_enc\
            .transform(df_test[onehot_before_columns])\
            .toarray()
        
    return df_train, df_test


def ce_cat_feat(df_train, df_test=None):    
    num_per_cat_dict_train = {}
    basic_cat_columns = [f'f_{i}' for i in range(2, 42)]
    group_columns = [
        column for column in df_train.columns
        if column.startswith('gp_cnt')
    ]
    all_cat_columns = basic_cat_columns + group_columns
    
    for feat_name in all_cat_columns:
        num_per_cat_dict_train[feat_name] = df_train[feat_name].nunique()
    
    ce_columns = sorted([
        key for key, value in num_per_cat_dict_train.items() 
        if 3 <= value
    ])
    ce_after_columns = list(map(lambda x: f'ce_{x}', ce_columns))
    
    # count_enc = CountEncoder(cols=ce_columns, normalize=True)
    count_enc = CountEncoder(cols=ce_columns)
    
    if df_test is not None:
        df_all = pd.concat((df_train, df_test), ignore_index=True)
        count_enc.fit(df_all[ce_columns])
        del df_all
    else:
        count_enc.fit(df_train[ce_columns])
    
    df_train[ce_after_columns] = count_enc.transform(df_train[ce_columns])
    if df_test is not None:
        df_test[ce_after_columns] = count_enc.transform(df_test[ce_columns])
        
    return df_train, df_test


def te_cat_feat(df_train, df_test=None):
    
    for label in ['is_clicked', 'is_installed']:
        if label == 'is_clicked':
            group_prefix = 'gp_click'
        else:
            group_prefix = 'gp_install'
            
        te_columns = [
            column for column in df_train.columns
            if column.startswith(group_prefix)
        ]
        
        if not te_columns: 
            continue
        
        te_after_columns = list(map(lambda x: f'te_{x}', te_columns))
        
        # te_cat_enc = preprocessing.TargetEncoder(random_state=2023)
        te_cat_enc = TargetEncoder(cols=te_columns, min_samples_leaf=20, smoothing=10)
        
        df_train[te_after_columns] = te_cat_enc.fit_transform(
            df_train[te_columns], df_train[label]
        )
        if df_test is not None:
            df_test[te_after_columns] = te_cat_enc.transform(df_test[te_columns])
            
    return df_train, df_test


def drop_cat_columns(df_train, df_test=None):
    group_columns = [
        column for column in df_train.columns
        if column.startswith('gp_click')
    ]
    exclude_columns = group_columns
    
    df_train = df_train.drop(exclude_columns, axis=1)
    if df_test is not None:
        df_test = df_test.drop(exclude_columns, axis=1)
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
    
    df_train, df_test = group_cat_feat(df_train, df_test)
    
    df_train, df_test = te_cat_feat(df_train, df_test)
        
    df_train, df_test = ce_cat_feat(df_train, df_test)
    
    if IS_CATEGORIFY:
        df_train, df_test = categorify_cat_feat(df_train, df_test)
    
    # df_train, df_test = onehot_cat_feat(df_train, df_test)
    
    df_train, df_test = drop_cat_columns(df_train, df_test)
        
    return df_train, df_test


def min_max_dense_feat(df_train, df_test=None):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    dense_columns = [f'f_{i}' for i in range(33, 80)] + \
        [column for column in df_train.columns if column.startswith('ce_')] + \
        [column for column in df_train.columns if column.startswith('te_')]

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
    if IS_PROCESS_DENSE:
        print('processing dense feature')
        df_train, df_test = process_dense_feat(df_train, df_test)
    
    return df_train, df_test


def save_output_df(df_train, df_test, test_date, output_dir):
    print('after process shape')
    print(df_train.shape, df_test.shape)

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
        [column for column in df_train.columns if column.startswith('ce_')] + \
        [column for column in df_train.columns if column.startswith('te_')]
    
    cat_columns = [f'f_{i}' for i in range(2, 7)] + \
        [f'f_{i}' for i in range(8, 33)] + \
        [column for column in df_train.columns if column.startswith('gp_cnt')]

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
    IS_CATEGORIFY = True
    IS_PROCESS_DENSE = True
    IS_COMBINE = False
    TEST_DATE = 60
    main(sys.argv[1:])


# python data/combine_recsys_2.py \
#    --input_dir '/home/vmagent/app/data/sharechat_recsys2023_data' \
#    --output_dir '/home/vmagent/app/data/recsys2023_process/raw10'