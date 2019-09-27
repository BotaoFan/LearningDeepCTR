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
    def __init_(self, feature_size, field_size,
                embedding_size=8, dropout_fm=[1, 1],
                deep_layers=[32, 32],  dropout_deep=[0.5, 0.5, 0.5],
                deep_layers_activation=tf.nn.relu,
                learning_rate=0.001, optimizer_type='adam',
                batch_norm=0, batch_norm_decay=0.995,
                verbose=False, random_seed=2016,
                use_fm=True, use_deep=True,
                loss_type='log_loss', eval_metric=roc_auc_score,
                l2_reg=0.0, greater_is_better=True):
        self.feature_size = feature_size  # M : number of all features(after one-hot)
        self.field_size = field_size  # F : number of all fields(before one-hot)
        self.embedding_size = embedding_size  # E : number of embedding
        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.l2_reg = l2_reg
        self.greater_is_better = greater_is_better
        self.verbose = verbose
        self.random_seed = random_seed
        self.weights = self._init_weights()

    def _weights(self):
        weights = dict()
        #embedding
        weights['embedding_weights'] = tf.Variable(tf.random.normal([self.feature_size, self.embedding_size], 0.0, 0.01),
                                                   name='embedding_weights')
        weights['embedding_bias'] = tf.Variable(tf.random.uniform([self.feature_size, 1], 0.0, 1.0),
                                                name='embedding_bias')
        #deep layers
        input_size = self.field_size * self.embedding_size + self.deep_layers[0]
        glorot = np.sqrt(2.0 / (input_size))
        weights['layer_0'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(input_size, self.deep_layers[0])))
        weights['bias_0'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(1, self.deep_layers[0])))
        for i in range(1, len(self.deep_layers)):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights['layer_%d' % i] = tf.Variable(
                np.random.normal(loc=0.0, scale=glorot, size=[self.deep_layers[i-1], self.deep_layers[i]], dtype=np.float32))
            weights['bias_%d' % i] = tf.Variable(
                np.random.normal(loc=0.0, scale=glorot, size=[1, self.deep_layers[i]], dtype=np.float32)
            )
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0 / input_size)
        weights['concat_weight'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(input_size, 1), dtype=np.float32))
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)
        return weights

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            #Initialize placeholder
            self.feat_index = tf.placeholder(tf.float32, shape=[None, None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')

            #Embedding
            self.embedding = tf.nn.embedding_lookup(self.weights['embedding_weights'], self.feat_index)
            self.embedding = tf.
































