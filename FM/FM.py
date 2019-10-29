#-*- coding:utf-8 -*-
# @Time : 2019/10/28
# @Author : Botao Fan
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from time import time


class FM(BaseEstimator):
    def __init__(self, num_param, embedding=8, learning_rate=0.001, l2_w=0.0, l2_v=0.0, echo=10, batch_size=256, random_seed=42):
        self.mum_param = num_param
        self.embedding = embedding
        self.learning_rate = learning_rate
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.echo = echo
        self.batch_size = batch_size
        self.random_seed = random_seed
        self._init_graph()

    def _init_weight(self):
        weight = dict()
        weight['w_bias'] = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='weight_bias')
        weight['w_linear'] = tf.Variable(tf.random_normal([1, self.mum_param]), dtype=tf.float32, name='weight_linear')
        weight['v'] = tf.Variable(tf.random_normal([self.embedding, self.mum_param]), dtype=tf.float32, name='factors')
        return weight

    def _init_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.x = tf.placeholder(tf.float32, shape=[None, self.mum_param], name='input_x')
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='input_y')
            weight = self._init_weight()
            self.linear_y = tf.add(tf.reduce_sum(tf.multiply(self.x, weight['w_linear']), 1, keepdims=True), weight['w_bias'])
            self.sum_square = tf.reduce_sum(tf.pow(tf.matmul(self.x, tf.transpose(weight['v'])), 2), 1, keepdims=True)
            self.square_sum = tf.reduce_sum(tf.matmul(tf.pow(self.x, 2), tf.transpose(tf.pow(weight['v'], 2))), 1, keepdims=True)
            self.quadratic_y = tf.multiply(0.5, tf.subtract(self.sum_square, self.square_sum))
            self.y_hat = tf.add(self.linear_y, self.quadratic_y)
            self.error = tf.reduce_mean(tf.square(self.y - self.y_hat))
            self.l2 = tf.add(
                tf.reduce_sum(tf.multiply(self.l2_w, tf.abs(weight['w_linear']))),
                tf.reduce_sum(tf.multiply(self.l2_v, tf.abs(weight['v'])))
            )
            self.loss = tf.add(self.error, self.l2)
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

    def _init_session(self):
        config = tf.ConfigProto(device_count={'gpu': 0})
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def _fit_on_batch(self, x_batch, y_batch):
        feed_dict = {self.x: x_batch,
                     self.y: y_batch
                     }
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, train_x, train_y, test_x=None, test_y=None):
        for i in range(self.echo):
            start_time = time()
            self._shuffle_x_y(train_x, train_y)
            n = len(train_x)
            batch = int(n / self.batch_size) + 1
            for j in range(batch):
                x_batch, y_batch = self._get_batch(train_x, train_y, self.batch_size, i)
                loss_batch = self._fit_on_batch(x_batch, y_batch)
            time_consume = time() - start_time
            train_loss = self._evaluate(train_x, train_y)
            if test_x is not None:
                test_loss = self._evaluate(test_x, test_y)
                print('[%d] Train mean loss: %.4f | Test mean loss: %.4f | Time consume is %.2f') % ((i + 1), train_loss, test_loss, time_consume)
            else:
                print('[%d] Train mean loss: %.4f | Time consume is %.2f') % ((i + 1), train_loss, time_consume)

    def _shuffle_x_y(self, x, y):
        rnd_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(rnd_state)
        np.random.shuffle(y)
        return None

    def _get_batch(self, x, y, batch_size, i):
        n = len(x)
        start = i * batch_size
        end = min(n, (i + 1) * batch_size)
        return x[start: end], y[start: end]

    def _evaluate(self, x, y):
        y_hat = self.predict(x)
        loss = mean_absolute_error(y, y_hat)
        return loss

    def predict(self, x):
        n = len(x)
        y_dummy = np.reshape([1] * n, [n, 1])
        batch = int(n / self.batch_size) + 1
        y_predict = None
        for i in range(batch):
            x_batch, y_dummy_batch = self._get_batch(x, y_dummy, self.batch_size, i)
            feed_dict = {self.x: x_batch,
                         self.y: y_dummy_batch}
            batch_out = self.sess.run(self.y_hat, feed_dict=feed_dict)
            if y_predict is None:
                y_predict = batch_out
            else:
                y_predict = np.concatenate((y_predict, batch_out))
        return y_predict


