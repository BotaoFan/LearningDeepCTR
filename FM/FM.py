#-*- coding:utf-8 -*-
# @Time : 2019/10/28
# @Author : Botao Fan
from sklearn.base import BaseEstimator
import tensorflow as tf

class FM(BaseEstimator):
    def __init__(self, m_param, embedding=8, learning_rate=0.001, l2_w=0.001, l2_v=0.001):
        self.m_param = m_param
        self.embedding = embedding
        self.learning_rate = learning_rate
        self.l2_w = l2_w
        self.l2_v = l2_v
        self._init_graph()

    def _init_weight(self):
        weight = dict()
        weight['w_bias'] = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='weight_bias')
        weight['w_linear'] = tf.Variable(tf.random_normal([self.m_param]), dtype=tf.float32, name='weight_linear')
        weight['v'] = tf.Variable(tf.random_normal([self.embedding, self.m_param]), dtype=tf.float32, name='factors')
        return weight

    def _init_graph(self):
        weight = self._init_weight()
        graph = tf.Graph()
        with graph.as_default():
            self.x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
            self.y = tf.placeholder(tf.int32, shape=[None, 1], name='input_y')
            self.linear_y = tf.add(tf.reduce_sum(tf.multiply(self.x, weight['w_linear']), 1, keepdims=True), weight['w_bias'])
            self.sum_square = tf.reduce_sum(tf.pow(tf.multiply(self.x, tf.transpose(weight['v'])), 2), 1, keepdims=True)
            self.square_sum = tf.reduce_sum(tf.multiply(tf.pow(self.x, 2), tf.transpose(tf.pow(weight['v'], 2))), 1, keepdims=True)
            self.quadratic_y = tf.multiply(0.5, tf.subtract(self.sum_square, self.square_sum))
            self.y_hat = tf.add(self.linear_y, self.quadratic_y)
            self.error = tf.reduce_mean(tf.square(self.y, self.y_hat))
            self.l2 = tf.add(
                tf.reduce_sum(tf.multiply(self.l2_w, weight['w_linear'])),
                tf.reduce_sum(tf.multiply(self.l2_v, weight['v']))
            )
            self.loss = self.add(self.error, self.l2)
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess = self._init_session()
            sess.run(init)


    def _init_session(self):
        config = tf.ConfigProto(device_count={'gpu': 0})
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def _fit_in_batch(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def eval_matrix(self):
        pass



