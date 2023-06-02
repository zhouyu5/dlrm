# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN, PredictionLayer
from deepctr_torch.inputs import combined_dnn_input

from .mmoe2 import MMOE2



class MMOE3(MMOE2):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_experts: integer, number of experts.
    :param expert_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param gate_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part.
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression'].
    :param task_names: list of str, indicating the predict target of each tasks.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128),
                 gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(64,), l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(MMOE3, self).__init__(dnn_feature_columns, num_experts, expert_dnn_hidden_units,
                 gate_dnn_hidden_units, tower_dnn_hidden_units, l2_reg_linear,
                 l2_reg_embedding, l2_reg_dnn,
                 init_std, seed, dnn_dropout, dnn_activation, dnn_use_bn,
                 task_types, task_names, device, gpus)
        
        # tower dnn (task-specific)
        self.tower_dnn = nn.ModuleList(
            [DNN(expert_dnn_hidden_units[-1] + tower_dnn_hidden_units[-1], 
                    tower_dnn_hidden_units, activation=dnn_activation,
                    l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                    init_std=init_std, device=device) for _ in range(self.num_tasks)]
        )
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
            l2=l2_reg_dnn)
        
        self.to(device)
        

    def get_tower_output(self, X):
        mmoe_outs = self.gen_mmoe_output(X)

        # tower dnn (task-specific)
        tower_dnn_outs = []
        for i in range(self.num_tasks):
            if i > 0:
                mmoe_outs[i] = torch.cat(
                    (mmoe_outs[i], aux_hidden_output), 
                    -1
                )
            tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
            tower_dnn_outs.append(tower_dnn_out)
            if i == 0:
                aux_hidden_output = tower_dnn_out.clone().detach()
        
        tower_dnn_outs = torch.cat(tower_dnn_outs, -1)
        return tower_dnn_outs


    def gen_cat_emb(self, X):
        pass


    def forward(self, X):
        mmoe_outs = self.gen_mmoe_output(X)

        # tower dnn (task-specific)
        task_outs = []
        aux_hidden_outputs = []
        for i in range(self.num_tasks):
            
            tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
            tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
            aux_hidden_outputs.append(tower_dnn_out.clone().detach())

        task_outs = []
        for i in range(self.num_tasks):
            mmoe_outs[i] = torch.cat(
                (mmoe_outs[i], aux_hidden_outputs[1-i]), 
                -1
            )
            tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
            tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        
        task_outs = torch.cat(task_outs, -1)
        return task_outs
