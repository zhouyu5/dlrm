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
from models import MMOE2


class EvaluatingCallback(keras.callbacks.Callback):
    def __init__(self, label_list, feature_names,
                 train=None, valid=None, test=None, 
                 pred_save_path=None, emb_save_dir=None,
                ):
        self.label_list = label_list
        self.feature_names = feature_names
        self.valid = valid
        self.test = test
        self.val_model_input = None
        if valid is not None:
            self.val_model_input = {name: valid[name] for name in feature_names}
        self.test_model_input = None
        self.all_data = train
        if test is not None:
            self.all_data = pd.concat((train, test), ignore_index=True)
            self.test_model_input = {name: test[name] for name in feature_names}
        self.pred_save_path = pred_save_path
        self.emb_save_dir = emb_save_dir

    def save_predict(self, test, test_model_input):
        print(f'begin save predictions...')
        target = self.label_list
        i = target.index('is_installed')
        pred_ans = self.model.predict(test_model_input, 256)
        num_samples = pred_ans.shape[0]
        pd.DataFrame({
            'RowId': test['f_0'].values,
            'is_clicked': [0.0] * num_samples,
            'is_installed': pred_ans[:, i].round(decimals=5)
        }).to_csv(self.pred_save_path, sep='\t', header=True, index=False)

    def evaluate_valid(self):
        def H(p):
            return -p*math.log(p) - (1-p)*math.log(1-p)
        valid = self.valid
        val_model_input = self.val_model_input
        target = self.label_list
        pred_ans = self.model.predict(val_model_input, 256)
        pred_ans = pred_ans.round(decimals=5)
        num_samples = pred_ans.shape[0]
        print(f'total number of samples in valid set: {num_samples}')
        for i, target_name in enumerate(target):
            bce_loss = log_loss(valid[target[i]].values, pred_ans[:, i])
            XTR = valid[target[i]].sum() / valid[target[i]].shape[0]
            nce_loss = bce_loss / H(XTR)
            print("%s valid NCELoss" % target_name, round(nce_loss, 4))
            print("%s valid LogLoss" % target_name, round(bce_loss, 4))
            print("%s valid AUC" % target_name, round(roc_auc_score(valid[target[i]].values, pred_ans[:, i]), 4))
        
    def on_epoch_end(self, epoch, logs=None):
        if self.valid is not None:
            self.evaluate_valid()
        if self.pred_save_path:
            self.save_predict(self.test, self.test_model_input)
        print("########################################################")

    def export_embedding(self, row_id, model_input, emb_save_path):
        print(f'begin export embeddings to {emb_save_path}...')
        df = pd.DataFrame({
            'f_0': row_id,
        })
        record_emb_array = self.model.gen_record_emb(
            model_input, 256
        )
        emb_columns = [f'emb_{i}' for i in range(128)]
        df[emb_columns] = record_emb_array.round(decimals=5)
        df.to_csv(emb_save_path, sep='\t', header=True, index=False)
        print("########################################################")

    def on_train_end(self, logs=None):
        if self.emb_save_dir:
            for day in sorted(self.all_data['f_1'].unique()):
                data = self.all_data.loc[
                    self.all_data['f_1'] == day
                ]
                row_id = data['f_0'].values
                model_input = {name: data[name] for name in self.feature_names}
                emb_save_path = f'{emb_save_dir}/day_{day}.csv'
                self.export_embedding(row_id, model_input, emb_save_path)


class ConfigCallback(keras.callbacks.Callback):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
    
    def _get_optim(self):
        optimizer = self.optimizer
        lr = self.lr
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.model.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.model.parameters(), lr=lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.model.parameters(), lr=lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.model.parameters(), lr=lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim
    
    def on_train_begin(self, logs=None):
        print(f'You are choosing optimizer {self.optimizer}, initial lr: {self.lr}')
        self.model.optim = self._get_optim()


def get_exp_days_list(exp='single'):
    print(f'You are choosing exp: {exp}...')
    train_days_list = []
    val_days_list = []
    test_days_list = []

    if 'multi' in exp:
        train_days_list += [range(val_day-11, val_day) for val_day in range(60, 67)]
        # train_days_list += [range(0, val_day) for val_day in range(60, 67)]
        val_days_list += [[val_day] for val_day in range(60, 67)]
        test_days_list += [[val_day] for val_day in range(60, 67)]
    if 'day67' in exp:
        # stop_day_list = [6, 2, 14, 0, 1, 18, 5]
        # train_days_list += [[day for day in range(67) if day not in stop_day_list]]
        minus_day = 11
        train_days_list += [range(67-minus_day, 67)]
        val_days_list += [[66]]
        test_days_list += [[67]]
    if 'day60' in exp:
        # stop_day_list = [6, 2, 14, 0, 1, 18, 5]
        # train_days_list += [[day for day in range(22) if day not in stop_day_list]]
        train_days_list += [range(51, 60)]
        # train_days_list += [range(45, 60)]
        val_days_list += [[60]]
        test_days_list += [[60]]
    if 'last_week' in exp:
        train_days_list += [range(49, 60)]
        val_days_list += [range(60, 67)]
        test_days_list += [range(60, 67)]

    return train_days_list, val_days_list, test_days_list


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ########################################### 0. prepare params ###########################################
    # single, multi, last_week
    # exp_mode = 'multi,single'
    exp_mode = 'day60'
    train_days_list, val_days_list, test_days_list = get_exp_days_list(
        exp=exp_mode
    )

    for TRAIN_DAYS, VAL_DAYS, TEST_DAYS in zip(
        train_days_list, val_days_list, test_days_list):
        test_day = TEST_DAYS[-1]
        print(f'train_day: {TRAIN_DAYS}, val_days: {VAL_DAYS}')

        model_name = 'MMoE2' # MMoE, MMoE2, PLE
        
        input_data_dir = '/home/vmagent/app/data/recsys2023_process/raw11'
        save_dir = f'sub/{model_name}'
        pred_save_path = f'{save_dir}/sub_{model_name}_'\
            f'test-{test_day}.csv'
        emb_save_dir = f'{save_dir}/DNN_emb/test-{test_day}-short'
        # emb_save_dir = None
        shuffle = True

        tower_dnn_hidden_units=(64,)
        # tower_dnn_hidden_units=(64, 32)
        num_experts = 3
        embedding_dim = "auto"

        batch_size = 256
        epochs = 2
        # adagrad, adam, rmsprop
        optimizer = "adagrad"
        learning_rate = 1e-2
        l2_reg_linear = 0.0
        l2_reg_embedding = 0.0
        if save_dir:
            os.system(f'mkdir -p {save_dir}')
        if emb_save_dir:
            os.system(f'mkdir -p {emb_save_dir}')
        
        ########################################### 1. prepare data ###########################################
        # data format
        CAT_FEATURE_COUNT = 30
        INT_FEATURE_COUNT = 47

        sparse_features = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]
        dense_features = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
        target = 'is_installed,is_clicked,f_0,f_1'.split(',')

        train_data_path = [f'{input_data_dir}/train']
        val_data_path = [f'{input_data_dir}/valid']
        test_data_path = [f'{input_data_dir}/test']
        feat_colunms = target + dense_features + sparse_features
        train = pd.concat((pd.read_csv(f, sep='\t', names=feat_colunms) for f in train_data_path), ignore_index=True)
        if test_day < 67:
            valid = pd.concat((pd.read_csv(f, sep='\t', names=feat_colunms) for f in val_data_path), ignore_index=True)
        else:
            valid = None
        test = pd.concat((pd.read_csv(f, sep='\t', names=feat_colunms) for f in test_data_path), ignore_index=True)
        target = target[:-2]

        ### filter data
        train = train.loc[train['f_1'].isin(TRAIN_DAYS)]

        # 2.count #unique features for each sparse field,and record dense feature field name
        data = pd.concat((train, test), ignore_index=True)
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                                for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                for feat in dense_features]
        del data

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train_model_input = {name: train[name] for name in feature_names}
        
        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        if model_name == 'MMoE':
            model = MMOE(
                dnn_feature_columns, 
                num_experts=num_experts,
                tower_dnn_hidden_units=tower_dnn_hidden_units,
                task_types=['binary', 'binary'],
                l2_reg_linear=l2_reg_linear,
                l2_reg_embedding=l2_reg_embedding,
                task_names=target, device=device
            )
        elif model_name == 'MMoE2':
            model = MMOE2(
                dnn_feature_columns, 
                num_experts=num_experts,
                tower_dnn_hidden_units=tower_dnn_hidden_units,
                task_types=['binary', 'binary'],
                l2_reg_linear=l2_reg_linear,
                l2_reg_embedding=l2_reg_embedding,
                task_names=target, device=device
            )
        elif model_name == 'PLE':
            model = PLE(
                dnn_feature_columns, 
                num_levels=3,
                tower_dnn_hidden_units=tower_dnn_hidden_units,
                task_types=['binary', 'binary'],
                l2_reg_linear=l2_reg_linear,
                l2_reg_embedding=l2_reg_embedding,
                task_names=target, device=device
            )
        else:
            raise NotImplementedError

        model.compile(optimizer, loss=["binary_crossentropy", "binary_crossentropy"],
                    metrics=[], )
        
        evaluate_callback = EvaluatingCallback(
            target, feature_names,
            train, valid, test, 
            pred_save_path, emb_save_dir,
        )
        config_callback = ConfigCallback(
            optimizer, learning_rate
        )

        history = model.fit(
            train_model_input, train[target].values, 
            batch_size=batch_size, epochs=epochs, 
            # validation_data=(val_model_input, valid[target].values),
            shuffle=shuffle,
            callbacks=[
                evaluate_callback, config_callback
            ],
        )
    