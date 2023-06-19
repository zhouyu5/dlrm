# -*- coding:utf-8 -*-

import time
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList
from tqdm import tqdm

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN, PredictionLayer
from deepctr_torch.layers.utils import slice_arrays
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
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, num_classes=4,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(MMOE2, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std,
                                   seed=seed, device=device, gpus=gpus)
        self.num_tasks = len(task_names)

        assert len(tower_dnn_hidden_units) > 0, "tower_dnn_hidden_units "\
            "size must > 0"

        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ['binary', 'regression', 'multiclass']:
                raise ValueError("task must be binary or multiclass or regression, {} is illegal".format(task_type))

        self.num_experts = num_experts
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units

        # expert dnn
        self.expert_dnn = nn.ModuleList([DNN(self.input_dim, expert_dnn_hidden_units, activation=dnn_activation,
                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                             init_std=init_std, device=device) for _ in range(self.num_experts)])

        # gate dnn
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList([DNN(self.input_dim, gate_dnn_hidden_units, activation=dnn_activation,
                                               l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                               init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.gate_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim,
                       self.num_experts, bias=False) for _ in range(self.num_tasks)])

        # tower dnn (task-specific)
        self.tower_dnn = nn.ModuleList(
            [DNN(expert_dnn_hidden_units[-1], 
                    tower_dnn_hidden_units, activation=dnn_activation,
                    l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                    init_std=init_std, device=device)] + \
            [DNN(expert_dnn_hidden_units[-1] + tower_dnn_hidden_units[-1], 
                    tower_dnn_hidden_units, activation=dnn_activation,
                    l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                    init_std=init_std, device=device) for _ in range(self.num_tasks-1)]
        )
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
            l2=l2_reg_dnn)

        self.tower_dnn_final_layer = nn.ModuleList(
            [nn.Linear(tower_dnn_hidden_units[-1], num_classes, bias=False)] + \
            [nn.Linear(tower_dnn_hidden_units[-1], 1, bias=False)
                for _ in range(self.num_tasks-1)]
        )

        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])

        regularization_modules = [self.expert_dnn, self.gate_dnn_final_layer, self.tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

        def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
            """

            :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
                dictionary mapping input names to Numpy arrays.
            :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
            :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
            :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
            :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
            :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
            :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
            :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
            :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

            :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
            """
            if isinstance(x, dict):
                x = [x[feature] for feature in self.feature_index]

            do_validation = False
            if validation_data:
                do_validation = True
                if len(validation_data) == 2:
                    val_x, val_y = validation_data
                    val_sample_weight = None
                elif len(validation_data) == 3:
                    val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
                else:
                    raise ValueError(
                        'When passing a `validation_data` argument, '
                        'it must contain either 2 items (x_val, y_val), '
                        'or 3 items (x_val, y_val, val_sample_weights), '
                        'or alternatively it could be a dataset or a '
                        'dataset or a dataset iterator. '
                        'However we received `validation_data=%s`' % validation_data)
                if isinstance(val_x, dict):
                    val_x = [val_x[feature] for feature in self.feature_index]

            elif validation_split and 0. < validation_split < 1.:
                do_validation = True
                if hasattr(x[0], 'shape'):
                    split_at = int(x[0].shape[0] * (1. - validation_split))
                else:
                    split_at = int(len(x[0]) * (1. - validation_split))
                x, val_x = (slice_arrays(x, 0, split_at),
                            slice_arrays(x, split_at))
                y, val_y = (slice_arrays(y, 0, split_at),
                            slice_arrays(y, split_at))

            else:
                val_x = []
                val_y = []
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)

            train_tensor_data = Data.TensorDataset(
                torch.from_numpy(
                    np.concatenate(x, axis=-1)),
                torch.from_numpy(y))
            if batch_size is None:
                batch_size = 256

            model = self.train()
            loss_func = self.loss_func
            optim = self.optim

            if self.gpus:
                print('parallel running on these gpus:', self.gpus)
                model = torch.nn.DataParallel(model, device_ids=self.gpus)
                batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
            else:
                print(self.device)

            train_loader = DataLoader(
                dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

            sample_num = len(train_tensor_data)
            steps_per_epoch = (sample_num - 1) // batch_size + 1

            # configure callbacks
            callbacks = (callbacks or []) + [self.history]  # add history callback
            callbacks = CallbackList(callbacks)
            callbacks.set_model(self)
            callbacks.on_train_begin()
            callbacks.set_model(self)
            if not hasattr(callbacks, 'model'):  # for tf1.4
                callbacks.__setattr__('model', self)
            callbacks.model.stop_training = False

            # Train
            print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
                len(train_tensor_data), len(val_y), steps_per_epoch))
            for epoch in range(initial_epoch, epochs):
                callbacks.on_epoch_begin(epoch)
                epoch_logs = {}
                start_time = time.time()
                loss_epoch = 0
                total_loss_epoch = 0
                train_result = {}
                try:
                    with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                        for _, (x_train, y_train) in t:
                            x = x_train.to(self.device).float()
                            y = y_train.to(self.device).float()

                            y_pred = model(x).squeeze()

                            optim.zero_grad()
                            if isinstance(loss_func, list):
                                assert len(loss_func) == self.num_tasks,\
                                    "the length of `loss_func` should be equal with `self.num_tasks`"
                                loss = sum(
                                    [loss_func[i](y_pred[:, i:], y[:, i], reduction='sum') for i in range(self.num_tasks)])
                            else:
                                loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                            reg_loss = self.get_regularization_loss()

                            total_loss = loss + reg_loss + self.aux_loss

                            loss_epoch += loss.item()
                            total_loss_epoch += total_loss.item()
                            total_loss.backward()
                            optim.step()

                            if verbose > 0:
                                for name, metric_fun in self.metrics.items():
                                    if name not in train_result:
                                        train_result[name] = []
                                    train_result[name].append(metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))


                except KeyboardInterrupt:
                    t.close()
                    raise
                t.close()

                # Add epoch_logs
                epoch_logs["loss"] = total_loss_epoch / sample_num
                for name, result in train_result.items():
                    epoch_logs[name] = np.sum(result) / steps_per_epoch

                if do_validation:
                    eval_result = self.evaluate(val_x, val_y, batch_size)
                    for name, result in eval_result.items():
                        epoch_logs["val_" + name] = result
                # verbose
                if verbose > 0:
                    epoch_time = int(time.time() - start_time)
                    print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                    eval_str = "{0}s - loss: {1: .4f}".format(
                        epoch_time, epoch_logs["loss"])

                    for name in self.metrics:
                        eval_str += " - " + name + \
                                    ": {0: .4f}".format(epoch_logs[name])

                    if do_validation:
                        for name in self.metrics:
                            eval_str += " - " + "val_" + name + \
                                        ": {0: .4f}".format(epoch_logs["val_" + name])
                    print(eval_str)
                callbacks.on_epoch_end(epoch, epoch_logs)
                if self.stop_training:
                    break

            callbacks.on_train_end()

            return self.history
