# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
import sys
import os
import math
from tensorflow import keras
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
from models import MMOE2, MMOE3


class RunningCallback(keras.callbacks.Callback):
    def __init__(self, label_list, feature_names,
                 cat_feature_columns,
                 train=None, valid=None, test=None, 
                 pred_save_path=None, emb_save_dir=None, 
                 model_save_path=None, model_load_path=None
                ):
        self.label_list = label_list
        self.feature_names = feature_names
        self.cat_feature_columns = cat_feature_columns
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
        self.model_save_path = model_save_path
        self.model_load_path = model_load_path

    def save_predict(self, test, test_model_input, epoch):
        if not self.pred_save_path:
            return
        print(f'begin save predictions...')
        target = self.label_list
        i = target.index('is_installed')
        pred_ans = self.model.predict(test_model_input, 256)
        num_samples = pred_ans.shape[0]
        pd.DataFrame({
            'RowId': test['f_0'].values,
            'is_clicked': [0.0] * num_samples,
            'is_installed': pred_ans[:, i].round(decimals=5)
        }).to_csv(f'{self.pred_save_path}-ep{epoch}.csv', sep='\t', header=True, index=False)

    def evaluate_valid(self):
        def H(p):
            return -p*math.log(p) - (1-p)*math.log(1-p)
        if self.valid is None:
            return
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
        
    def save_model(self, epoch):
        if not self.model_save_path:
            return
        print(f'begin save model of epoch {epoch}...')
        model_save_path = f'{self.model_save_path}-ep{epoch}.pkl'
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.model.optim.state_dict()
        }
        torch.save(state, model_save_path)
    
    def on_epoch_end(self, epoch, logs=None):
        self.evaluate_valid()
        self.save_predict(self.test, self.test_model_input, epoch)
        self.save_model(epoch)
        self.scheduler.step()
        print("########################################################")
    
    def load_model(self, continue_train=False):
        last_epoch = -1
        if not self.model_load_path:
            return last_epoch
        print(f'load exist model from {self.model_load_path}...')
        checkpoint = torch.load(self.model_load_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if continue_train:
            self.model.optim.load_state_dict(checkpoint['optimizer'])
            last_epoch = checkpoint['epoch']
        return last_epoch
    
    def set_lr_scheduler(self, last_epoch):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.model.optim, step_size=10, gamma=0.1, last_epoch=last_epoch
        )

    def on_train_begin(self, logs=None):
        last_epoch = self.load_model()
        self.set_lr_scheduler(last_epoch)

    def export_embedding(self, row_id, model_input, emb_save_path):
        print(f'begin export embeddings to {emb_save_path}...')
        df = pd.DataFrame({
            'f_0': row_id,
        })

        # record_emb_array = self.model.gen_record_emb(
        #     model_input, 256
        # )
        # emb_columns = [f'emb_{i}' for i in range(128)]

        record_emb_array = self.model.gen_cat_emb(
            model_input, self.cat_feature_columns, 256
        )
        cat_feat_emb_dim = sum([feat.embedding_dim for feat in self.cat_feature_columns])
        emb_columns = [f'emb_{i}' for i in range(cat_feat_emb_dim)]
        print(f'\t total cat feature emb dim is {cat_feat_emb_dim}')

        df[emb_columns] = record_emb_array.round(decimals=5)
        df.to_csv(emb_save_path, sep='\t', header=True, index=False)

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
        print("########################################################")


class CustomizeCompileModel():
    def __init__(self, model, optimizer, lr, loss, loss_weight_list, weight_decay=0):
        self.model = model
        self.model.optim = self._get_optim(optimizer, lr, weight_decay)
        self.model.loss_func = self._get_loss_func(loss, loss_weight_list)

    def fix_weight_decay(self):
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
                no_decay.append(param)
            else:
                decay.append(param)
        params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
        return params
    
    def _get_optim(self, optimizer, lr, weight_decay):
        print(f'You are choosing optimizer {optimizer}, initial lr: {lr}, weight decay: {weight_decay}')
        if weight_decay > 0:
            params = self.fix_weight_decay()
        else:
            params = self.model.parameters()
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
            elif optimizer == "adam":
                optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim
    
    def _get_loss_func_single(self, loss):
        if loss == "binary_crossentropy":
            loss_func = F.binary_cross_entropy
        elif loss == 'weight_bce':
            loss_func_temp = F.binary_cross_entropy
            loss_func = lambda *args, **kwargs: loss_func_temp(
                *args, weight=torch.where(args[1] > 0, 1.0, 1.5),
                reduction=kwargs['reduction']
            )
        elif loss == "mse":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func
    
    def _get_loss_func(self, loss, loss_weight_list):        
        print(f'You are choosing loss function: {loss}, loss weight: {loss_weight_list}')
        
        if isinstance(loss, str):
            loss_func = self._get_loss_func_single(loss)
        elif isinstance(loss, list):
            loss_func = [
                lambda *args, **kwargs: self._get_loss_func_single(loss_single)(*args, **kwargs) * loss_weight 
                for loss_single, loss_weight in zip(loss, loss_weight_list)
            ]
        else:
            loss_func = loss
        return loss_func


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
        # train_days_list += [range(45, 67)]
        val_days_list += [[66]]
        test_days_list += [[67]]
    if 'day60' in exp:
        # stop_day_list = [6, 2, 14, 0, 1, 18, 5]
        # train_days_list += [[day for day in range(22) if day not in stop_day_list]]
        # train_days_list += [range(51, 60)]
        train_days_list += [range(45, 60)]
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

        ########################################### 0. set params ###########################################
        model_name = 'MMoE2' # MMoE, MMoE2, PLE
        # loss = ["binary_crossentropy", "binary_crossentropy"]
        loss = ["weight_bce", "weight_bce"]
        num_tasks = len(loss)
        loss_weight_list = [1] * num_tasks
        # weight_decay = 1e-4
        weight_decay = 0
        
        input_data_dir = '/home/vmagent/app/data/recsys2023_process/raw19'
        print(f'input data path: {input_data_dir}')
        save_dir = f'sub/{model_name}'
        pred_save_dir = f'{save_dir}/pred-{exp_mode}'
        pred_save_path = f'{pred_save_dir}/sub-{model_name}'
        model_save_dir = f'{save_dir}/model-{exp_mode}'
        model_save_path = f'{model_save_dir}/{model_name}'
        # model_load_path = f'sub/MMoE2/model-{exp_mode}-all/MMoE2-ep4.pkl'
        # model_load_path = f'sub/MMoE2/model-{exp_mode}/MMoE2-ep7.pkl'
        model_load_path = None
        # emb_save_dir = f'{save_dir}/DNN_cat_emb-{exp_mode}'
        emb_save_dir = None
        shuffle = False

        tower_dnn_hidden_units=(64,)
        # tower_dnn_hidden_units=(64, 32)
        num_experts = num_tasks + 1
        embedding_dim = "auto"

        batch_size = 64
        epochs = 5
        # adagrad, adam, rmsprop
        optimizer = "adagrad"
        learning_rate = 0.01
        l2_reg_linear = 0.01
        l2_reg_embedding = 0.01
        if pred_save_dir:
            os.system(f'mkdir -p {pred_save_dir}')
        if model_save_dir:
            os.system(f'mkdir -p {model_save_dir}')
        if emb_save_dir:
            os.system(f'mkdir -p {emb_save_dir}')
        
        ########################################### 1. prepare data ###########################################
        # data format
        CAT_FEATURE_COUNT = 30
        INT_FEATURE_COUNT = 47

        sparse_features = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]
        dense_features = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
        target = 'is_installed,is_clicked,new_label,f_0,f_1'.split(',')

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
        target = target[:-3]

        ########################################### 2. prepare emb dim and field name ###########################################
        # data = pd.concat((train, test), ignore_index=True)
        data = train.copy()
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                                for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                for feat in dense_features]
        cat_feature_columns = [
            SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
            for feat in ['cat_0', 'cat_2', 'cat_4', 'cat_10', 'cat_12', 'cat_15']
        ]
        del data

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        ########################################### 3. generate train input data for model ###########################################
        ### filter data
        train = train.loc[train['f_1'].isin(TRAIN_DAYS)]
        train_model_input = {name: train[name] for name in feature_names}
        
        ########################################### 4. Define Model,train,predict and evaluate ###########################################
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        # TODO: find class by name
        if model_name == 'MMoE':
            model = MMOE(
                dnn_feature_columns, 
                num_experts=num_experts,
                tower_dnn_hidden_units=tower_dnn_hidden_units,
                task_types=['binary'] * num_tasks,
                l2_reg_linear=l2_reg_linear,
                l2_reg_embedding=l2_reg_embedding,
                task_names=target, device=device
            )
        elif model_name == 'MMoE2':
            model = MMOE2(
                dnn_feature_columns, 
                num_experts=num_experts,
                tower_dnn_hidden_units=tower_dnn_hidden_units,
                task_types=['binary'] * num_tasks,
                l2_reg_linear=l2_reg_linear,
                l2_reg_embedding=l2_reg_embedding,
                task_names=target, device=device
            )
        elif model_name == 'MMoE3':
            model = MMOE3(
                dnn_feature_columns, 
                num_experts=num_experts,
                tower_dnn_hidden_units=tower_dnn_hidden_units,
                task_types=['binary'] * num_tasks,
                l2_reg_linear=l2_reg_linear,
                l2_reg_embedding=l2_reg_embedding,
                task_names=target, device=device
            )
        elif model_name == 'PLE':
            model = PLE(
                dnn_feature_columns, 
                num_levels=3,
                tower_dnn_hidden_units=tower_dnn_hidden_units,
                task_types=['binary'] * num_tasks,
                l2_reg_linear=l2_reg_linear,
                l2_reg_embedding=l2_reg_embedding,
                task_names=target, device=device
            )
        else:
            raise NotImplementedError

        model.compile(optimizer, loss=["binary_crossentropy"] * num_tasks)
        CustomizeCompileModel(
            model, optimizer, learning_rate, loss, loss_weight_list, 
            weight_decay=weight_decay
        )
        
        evaluate_callback = RunningCallback(
            target, feature_names,
            cat_feature_columns,
            train, valid, test, 
            pred_save_path, emb_save_dir,
            model_save_path, model_load_path,
        )

        history = model.fit(
            train_model_input, train[target].values, 
            batch_size=batch_size, epochs=epochs, 
            # validation_data=(val_model_input, valid[target].values),
            shuffle=shuffle,
            callbacks=[
                evaluate_callback, 
            ],
        )
    