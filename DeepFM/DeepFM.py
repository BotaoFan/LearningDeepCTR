#-*- coding:utf-8 -*-
# @Time : 2019/9/26
# @Author : Botao Fan

import tensorflow as tf
import pandas as pd
import numpy as np

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
                 embedding_size=8, drop_fm=[1.0, 1.0],
                 deep_layers=[32, 32], drop_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type='adam',
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=42,
                 use_fm=True,use_deep=True,
                 loss_type='logloss', eval_metric=roc_auc_score,
                 l2_reg=0, greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ['logloss', 'mse'], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

