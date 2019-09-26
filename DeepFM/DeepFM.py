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
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.drop_fm = drop_fm
        self.drop_deep = drop_deep
        self.deep_layers = deep_layers
        self.deep_layers_activation=deep_layers_activation
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.verbose = verbose
        self.random_seed = random_seed
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.l2_reg = l2_reg
        self.greater_is_better = greater_is_better

        self._init_graph()

    def _init_graph(self):
        pass

    def _initialize_weight(self):
        weights = dict()
        #FM Layer
        weights['feature_embedding'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name='feature_embedding')
        weights['feature_bias'] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 0.01),
            name='feature_bias'
        )
        #Deep Layers
        layers_num = len(self.deep_layers)
        input_num = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_num + self.deep_layers[0]))
        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0.0, scale=glorot, size=[input_num, self.deep_layers[0]], dtype=np.float32))
        weights['bias_0'] = tf.Variable(
            np.random.normal(loc=0.0, scale=glorot, size=[1, self.deep_layers[0]], dtype=np.float32)
        )

        for i in range(1, layers_num):
            glorot = np.sqrt(2.0 / (layers_num[i] + layers_num[i - 1]))
            weights['layer_%d' % i] = tf.Variable(
                np.random.normal(loc=0.0, scale=glorot, size=[layers_num[i-1], layers_num[i]], dtype=np.float32))
            weights['bias_%d' % i] = tf.Variable(
                np.random.normal(loc=0.0, scale=glorot, size=[1, self.deep_layers[i]], dtype=np.float32)
            )



























