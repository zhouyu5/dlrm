# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
import sys
import os
import math
from tensorflow import keras
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

from data.recsys import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    LABEL_NAME,
)


class EvaluatingCallback(keras.callbacks.Callback):
    def save_predict(self, epoch):
        print(f'begin save predictions for epoch {epoch}...')
        i = target.index('is_installed')
        labels = test[target[i]].values
        pred_ans = self.model.predict(test_model_input, 256)
        num_samples = pred_ans.shape[0]
        save_path = f'{save_dir}sub_{epoch}.csv'
        pd.DataFrame({
            'row_id': labels,
            'is_clicked': [0.0] * num_samples,
            'is_installed': pred_ans[:, i]
        }).to_csv(save_path, sep='\t', header=True, index=False)
        
    def on_epoch_end(self, epoch, logs=None):
        def H(p):
            return -p*math.log(p) - (1-p)*math.log(1-p)
        pred_ans = self.model.predict(val_model_input, 256)
        num_samples = pred_ans.shape[0]
        print(f'total number of samples in valid set: {num_samples}')
        for i, target_name in enumerate(target):
            bce_loss = log_loss(valid[target[i]].values, pred_ans[:, i])
            XTR = valid[target[i]].sum() / valid[target[i]].shape[0]
            nce_loss = bce_loss / H(XTR)
            print("%s valid NCELoss" % target_name, round(nce_loss, 4))
            print("%s valid LogLoss" % target_name, round(bce_loss, 4))
            print("%s valid AUC" % target_name, round(roc_auc_score(valid[target[i]].values, pred_ans[:, i]), 4))
        if is_save_predict:
            self.save_predict(epoch=epoch)
        print("########################################################")


def get_exp_days_list(exp='single'):
    print(f'You are choosing exp: {exp}...')
    if exp == 'single':
        train_days_list = [range(11, 22)]
        val_days_list = [range(21, 22)]
    elif exp == 'multi':
        train_days_list = [range(val_day-11, val_day) for val_day in range(15, 22)]
        val_days_list = [[val_day] for val_day in range(15, 22)]
    elif exp == 'last-week':
        train_days_list = [range(4, 15)]
        val_days_list = [range(15, 22)]
    else:
        raise NotImplementedError

    test_days_list = [range(22, 23)] * len(train_days_list)

    return train_days_list, val_days_list, test_days_list


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ########################################### 0. prepare params ###########################################
    # day60: 15, day61: 16, day 62: 17, day 63: 18, day 64: 19, day 65: 20, day66: 21
    train_days_list, val_days_list, test_days_list = get_exp_days_list(
        # exp='single',
        # exp='multi',
        exp='last-week',
    )

    for TRAIN_DAYS, VAL_DAYS, TEST_DAYS in zip(
        train_days_list, val_days_list, test_days_list):
        print(f'train_day: {TRAIN_DAYS}, val_days: {VAL_DAYS}')

        is_save_predict = False
        num_experts = 3
        batch_size = 256
        epochs = 1
        # adagrad, adam, rmsprop
        optimizer = "adagrad"
        # learning_rate = 1e-2
        shuffle = True
        save_dir = 'predict/raw2_'
        input_data_dir = '/home/vmagent/app/data/recsys2023_process/raw2'
        l2_reg_linear = 0.0
        l2_reg_embedding = 0.0

        sparse_features = DEFAULT_CAT_NAMES
        dense_features = DEFAULT_INT_NAMES
        target = LABEL_NAME.split(',')

        # 1. prepare data
        train_data_path = [f'{input_data_dir}/day_{i}' for i in TRAIN_DAYS]
        val_data_path = [f'{input_data_dir}/day_{i}' for i in VAL_DAYS]
        test_data_path = [f'{input_data_dir}/day_{i}' for i in TEST_DAYS]
        feat_colunms = target + DEFAULT_INT_NAMES + DEFAULT_CAT_NAMES
        train = pd.concat((pd.read_csv(f, sep='\t', names=feat_colunms) for f in train_data_path), ignore_index=True)
        valid = pd.concat((pd.read_csv(f, sep='\t', names=feat_colunms) for f in val_data_path), ignore_index=True)
        test = pd.concat((pd.read_csv(f, sep='\t', names=feat_colunms) for f in test_data_path), ignore_index=True)

        # 2.count #unique features for each sparse field,and record dense feature field name
        data = pd.concat((train, valid, test), ignore_index=True)
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=16)
                                for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                for feat in dense_features]
        del data

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train_model_input = {name: train[name] for name in feature_names}
        val_model_input = {name: valid[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        model = MMOE(
            dnn_feature_columns, 
            num_experts=num_experts,
            task_types=['binary', 'binary'],
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            task_names=target, device=device
        )
        model.compile(optimizer, loss=["binary_crossentropy", "binary_crossentropy"],
                    metrics=['binary_crossentropy'], )

        history = model.fit(
            train_model_input, train[target].values, 
            batch_size=batch_size, epochs=epochs, 
            # validation_data=(val_model_input, valid[target].values),
            shuffle=shuffle,
            callbacks=[EvaluatingCallback()],
        )
    