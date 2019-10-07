#-*- coding:utf-8 -*-
# @Time : 2019/10/6
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


class DeepCrossNetwork(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
                embedding_size=8, cross_layer_num=3, dropout_linear=1.0,
                deep_layers=(32, 32),  dropout_deep=(0.5, 0.5, 0.5),
                deep_layers_activation=tf.nn.relu,
                epoch=10, batch_size=256,
                learning_rate=0.001, optimizer_type='adam',
                batch_norm=0, batch_norm_decay=0.995,
                verbose=True, random_seed=2016,
                use_cross=True, use_deep=True,
                loss_type='logloss', eval_metric=roc_auc_score,
                l2_reg=0.0, greater_is_better=True):
        assert (use_cross or use_deep)
        assert loss_type in ['logloss', 'mse'], \
            "loss_type can be either 'logloss' for classification or 'mse' for regression"
        self.feature_size = feature_size  # M : number of all features(after one-hot)
        self.field_size = field_size  # F : number of all fields(before one-hot)
        self.embedding_size = embedding_size  # E : number of embedding
        self.cross_layer_num = cross_layer_num
        self.deep_layers = deep_layers
        self.dropout_linear = dropout_linear
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_cross = use_cross
        self.eval_metric = eval_metric
        self.use_deep = use_deep
        self.loss_type = loss_type
        self.l2_reg = l2_reg
        self.greater_is_better = greater_is_better
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.verbose = verbose
        self.random_seed = random_seed
        self.train_result, self.valid_result = [], []
        self._init_graph()

    def _init_weight(self):
        weight = dict()
        weight['embedding_weight'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), dtype=np.float32)
        weight['embedding_bias'] = tf.Variable(
            tf.random_normal([self.feature_size, 1], 0.0, 0.01), dtype=np.float32)
        #Cross layers weights
        cross_layer_size = self.embedding_size * self.field_size
        glorot = np.sqrt(1.0 / cross_layer_size)
        for i in range(self.cross_layer_num):
            weight['cross_%d' % i] = tf.Variable(
                tf.random_normal([cross_layer_size, 1], 0.0, glorot), dtype=np.float32)
            weight['cross_bias_%d' % i] = tf.Variable(
                tf.random_normal([cross_layer_size, 1], 0.0, glorot), dtype=np.float32)
        #MLP weights
        glorot = np.sqrt(2.0 / (self.field_size * self.embedding_size + self.deep_layers[0]))
        weight['layer_0'] = tf.Variable(
            tf.random_normal([self.field_size * self.embedding_size, self.deep_layers[0]], 0.0, glorot), dtype=np.float32)
        weight['bias_0'] = tf.Variable(
            tf.random_normal([1, self.deep_layers[0]], 0.0, glorot), dtype=np.float32)

        for i in range(1, len(self.deep_layers)):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weight['layer_%d' % i] = tf.Variable(
                tf.random_normal([self.deep_layers[i-1], self.deep_layers[i]], 0.0, glorot), dtype=np.float32)
            weight['bias_%d' % i] = tf.Variable(
                tf.random_normal([1, self.deep_layers[i]], 0.0, 0.01), dtype=np.float32)
        #LR weights
        if self.use_cross and self.use_deep:
            lr_input_size = self.field_size + self.embedding_size * self.field_size + self.deep_layers[-1]
        elif self.use_cross:
            lr_input_size = self.field_size + self.embedding_size * self.field_size
        else:
            lr_input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0 / (lr_input_size + 1))
        weight['lr_weight'] = tf.Variable(tf.random_normal([lr_input_size, 1], 0.0, glorot), dtype=np.float32)
        weight['lr_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)
        return weight


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.weight = self._init_weight()
            tf.set_random_seed(self.random_seed)
            self.feat_idx = tf.placeholder(dtype=np.int32, shape=[None, None], name='feat_index')
            self.feat_val = tf.placeholder(dtype=np.float32, shape=[None, None], name='feat_value')
            self.label = tf.placeholder(dtype=np.float32, shape=[None, 1], name='label')
            self.dropout_keep_linear = tf.placeholder(dtype=np.float32, shape=[None], name='dropout_keep_linear')
            self.dropout_keep_deep = tf.placeholder(dtype=np.float32, shape=[None], name='dropout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            #model
            feat_val = tf.reshape(self.feat_val, [-1, self.field_size, 1])
            self.embedding = tf.nn.embedding_lookup(self.weight['embedding_weight'], self.feat_idx)
            # first order
            self.first_order = tf.nn.embedding_lookup(self.weight['embedding_bias'], self.feat_idx)
            # 采用reduce_sum之后，会将行求和，然后去掉aixs 2，就将[[[1],[2],[3]],[[4],[5],[6]]]变为[[1,2,3],[4,5,6]]
            self.first_order = tf.reduce_sum(tf.multiply(self.first_order, feat_val), 2)
            self.first_order = tf.nn.dropout(self.first_order, self.dropout_keep_linear[0])
            # cross layer
            self.embedding = tf.multiply(self.embedding, feat_val)
            self.x0 = tf.reshape(self.embedding, [-1, self.field_size * self.embedding_size, 1])
            self.xl = tf.reshape(self.embedding, [-1, 1, self.field_size * self.embedding_size])
            for i in range(self.cross_layer_num):
                self.xl = tf.add(
                    tf.add(
                        tf.matmul(
                            tf.matmul(self.x0, self.xl),
                            self.weight['cross_%d' % i]),
                        self.weight['cross_bias_%d' % i]),
                    tf.reshape(self.xl, [-1, self.field_size * self.embedding_size, 1]))
                self.xl = tf.reshape(self.xl, [-1, 1, self.field_size * self.embedding_size])
            self.xl = tf.reshape(self.xl, [-1, self.field_size * self.embedding_size])
            self.cross_order = tf.nn.dropout(self.xl, self.dropout_keep_linear[0])
            # deep component
            self.deep_order = tf.reshape(self.embedding, [-1, self.field_size * self.embedding_size])
            self.deep_order = tf.nn.dropout(self.deep_order, self.dropout_keep_deep[0])
            for i in range(len(self.deep_layers)):
                deep_weights = self.weight['layer_%d' % i]
                deep_bias = self.weight['bias_%d' % i]
                self.deep_order = tf.add(tf.matmul(self.deep_order, deep_weights), deep_bias)
                self.deep_order = self.deep_layers_activation(self.deep_order)
                self.deep_order = tf.nn.dropout(self.deep_order, self.dropout_keep_deep[i + 1])
            # deep & cross network
            if self.use_cross and self.use_deep:
                self.lr_input = tf.concat([self.first_order, self.cross_order, self.deep_order], 1)
            elif self.use_cross:
                self.lr_input = tf.concat([self.first_order, self.cross_order], 1)
            else:
                self.lr_input = self.deep_order
            self.output = tf.add(tf.matmul(self.lr_input, self.weight['lr_weight']), self.weight['lr_bias'])
            # loss function
            if self.loss_type == 'logloss':
                self.output = tf.nn.sigmoid(self.output)
                self.loss = tf.losses.log_loss(self.label, self.output)
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.output))
            if self.l2_reg > 0:
                self.loss += self.l2_reg * tf.sqrt(tf.reduce_sum(tf.square(self.weight['lr_weight']), 0))
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += self.l2_reg * tf.sqrt(tf.reduce_sum(tf.square(self.weight['layer_%d' % i]), 0))

            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(
                    learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)
            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
            # number of parameters
            params_num = 0
            for variable in self.weight.values():
                shape = variable.get_shape()
                param_num = 1
                for dim in shape:
                    param_num *= dim
                params_num += param_num
            if self.verbose > 0:
                print ('#params: %d' % params_num)

    def _init_session(self):
        config = tf.ConfigProto(device_count={'gpu': 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def get_batch(self, xi, xv, y, index, batch_size):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        y_batch = y[start:end]
        y_batch = np.reshape(y_batch, [-1, 1])
        return xi[start:end], xv[start:end], y_batch

    def shuffle_three_list(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        return None

    def fit_on_batch(self, xi, xv, y):
        feed_dict = {self.feat_idx: xi,
                     self.feat_val: xv,
                     self.label: y,
                     self.dropout_keep_linear: self.dropout_linear,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, xi_train, xv_train, y_train,
            xi_valid=None, xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        has_valid = False if xi_valid is None else True
        train_results = []
        valid_results = []
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_three_list(xi_train, xv_train, y_train)
            total_batch = int(np.ceil(len(y_train) / self.batch_size))
            for i in range(total_batch):
                xi_batch, xv_batch, y_batch = self.get_batch(xi_train, xv_train, y_train,
                                                             i, self.batch_size)
                self.fit_on_batch(xi_batch, xv_batch, y_batch)
            train_result = self.evaluate(xi_train, xv_train, y_train)
            train_results.append(train_result)
            if has_valid:
                valid_result = self.evaluate(xi_valid, xv_valid, y_valid)
                valid_results.append(valid_result)
            if self.verbose:
                if has_valid:
                    print('[%d] train-result = %.4f, valid-result = %.4f [%.1f s]' %
                          (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print('[%d] train-result = %.4f, [%.1f s]' %
                              (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                print('Conditions of Early Termination Have Achieved!')
                break
        return None

    def predict(self, xi, xv):
        dummy_y = [1] * len(xi)
        batch_index = 0
        xi_batch, xv_batch, y_batch = self.get_batch(xi, xv, dummy_y, batch_index, self.batch_size)
        y_pred = None
        while len(xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_idx: xi_batch,
                         self.feat_val: xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_linear: [1.0],
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.output, feed_dict=feed_dict)
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
            batch_index += 1
            xi_batch, xv_batch, y_batch = self.get_batch(xi, xv, dummy_y, batch_index, self.batch_size,)
        return y_pred

    def evaluate(self, xi, xv, y):
        y_pred = self.predict(xi, xv)
        return self.eval_metric(y, y_pred)

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] < valid_result[-3] \
                        < valid_result[-4] < valid_result[-5]:
                        return True
                else:
                    if valid_result[-1] > valid_result[-2] > valid_result[-3] \
                            > valid_result[-4] > valid_result[-5]:
                        return True
        return False
