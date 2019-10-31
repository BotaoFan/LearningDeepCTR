#-*- coding:utf-8 -*-
# @Time : 2019/10/30
# @Author : Botao Fan
from config import DATA_PATH
from data_prepare import data_prep

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, roc_auc_score
from time import time


class Linear(BaseEstimator):
    def __init__(self, param_size, field_size=2, l2_w=0,
                 learning_rate=0.001, epoch=10, batch_size=256, random_seed=42, evaluator=roc_auc_score):
        self.param_size = param_size
        self.field_size = field_size
        self.l2_w = l2_w
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.evaluator = evaluator

        self._init_graph()

    def _init_weight(self):
        weight = dict()
        weight['w_bias'] = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='weight_bias')
        weight['w_linear'] = tf.Variable(tf.random_normal([self.param_size, 1]), dtype=tf.float32, name='weight_linear')
        '''
        weight['v'] = tf.Variable(tf.random_normal([self.field_size, self.param_size, self.embedding_size]),
                                  dtype=tf.float32, name='hidden_factors')
        '''
        return weight

    def _init_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            weight = self._init_weight()
            self.idx = tf.placeholder(tf.int32, [None, self.field_size], 'index')
            self.val = tf.placeholder(tf.float32, [None, self.field_size], 'value')
            self.y = tf.placeholder(tf.float32, [None, 1], 'y')
            self.y_linear = tf.add(weight['w_bias'],
                                   tf.reduce_sum(
                                       tf.multiply(
                                           tf.squeeze(tf.nn.embedding_lookup(weight['w_linear'], self.idx)),
                                           self.val),
                                       1, keepdims=True)
                                   )
            '''
            self.v = []
            self.y_quadratic = None
            for i in range(self.field_size):
                self.v.append(tf.squeeze(tf.nn.embedding_lookup(weight['v'][i], tf.reshape(self.idx[:, i], [-1, 1]))))
            for i in range(self.field_size):
                for j in range(i, self.field_size):
                    if self.y_quadratic is None:
                        self.y_quadratic = tf.reduce_sum(
                            tf.multiply(
                                tf.multiply(self.v[j], tf.reshape(self.val[:, i], [-1, 1])),
                                tf.multiply(self.v[i], tf.reshape(self.val[:, j], [-1, 1]))
                            ), 1, keepdims=True)
                    else:
                        self.y_quadratic = tf.add(
                            tf.reduce_sum(
                                tf.multiply(
                                    tf.multiply(self.v[j], tf.reshape(self.val[:, i], [-1, 1])),
                                    tf.multiply(self.v[i], tf.reshape(self.val[:, j], [-1, 1]))
                                ), 1, keepdims=True),
                            self.y_quadratic)
            '''
            self.y_hat = self.y_linear
            self.error = tf.reduce_mean(tf.pow(tf.subtract(self.y, self.y_hat), 2))
            self.l2 = self.l2_w * tf.reduce_sum(weight['w_linear'])
            self.loss = self.l2 + self.error
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

    def _init_session(self):
        config = tf.ConfigProto(device_count={'gpu': 0})
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def _fit_on_batch(self, idx_batch, val_batch, y_batch):
        feed_dict = {
            self.idx: idx_batch,
            self.val: val_batch,
            self.y: y_batch
        }
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, idx, val, y, valid_idx=None, valid_val=None, valid_y=None):
        n = len(idx)
        batch = int(n / self.batch_size) + 1

        for epoch in range(self.epoch):
            start_time = time()
            self._shuffle_three_one_time(idx, val, y)
            for i in range(batch):
                idx_batch, val_batch, y_batch = self._get_batch(idx, val, y, i)
                loss_batch = self._fit_on_batch(idx_batch, val_batch, y_batch)
            time_consume = time() - start_time
            evaluate_loss_train = self._evaluate(idx, val, y)
            if valid_idx is not None:
                evaluate_loss_test = self._evaluate(valid_idx, valid_val, valid_y)
                print('[%d] Train mean loss: %.4f | Test mean loss: %.4f | Time consume is %.2f') %\
                     ((epoch + 1), evaluate_loss_train, evaluate_loss_test, time_consume)
            else:
                print('[%d] Train mean loss: %.4f |  Time consume is %.2f') %\
                     ((epoch + 1), evaluate_loss_train, time_consume)

    def _shuffle_three_one_time(self, idx, val, y):
        rnd_state = np.random.get_state()
        np.random.shuffle(idx)
        np.random.set_state(rnd_state)
        np.random.shuffle(val)
        np.random.set_state(rnd_state)
        np.random.shuffle(y)

    def _get_batch(self, idx, val, y, i):
        start = i * self.batch_size
        end = min((i + 1) * self.batch_size, len(idx))
        return idx[start: end], val[start: end], y[start: end]

    def predict(self, idx, val):
        n = len(idx)
        batch = int(n / self.batch_size) + 1
        y_dummy = np.reshape([1] * n, [-1, 1])
        y_hat = None
        for i in range(batch):
            idx_batch, val_batch, y_batch = self._get_batch(idx, val, y_dummy, i)
            feed_dict = {self.idx: idx_batch, self.val: val_batch, self.y: y_batch}
            y_hat_batch = self.sess.run(self.y_hat, feed_dict=feed_dict)
            if y_hat is None:
                y_hat = y_hat_batch
            else:
                y_hat = np.concatenate([y_hat, y_hat_batch])
        return y_hat

    def _evaluate(self, idx, val, y):
        y_hat = self.predict(idx, val)
        loss = self.evaluator(y, y_hat)
        return loss

