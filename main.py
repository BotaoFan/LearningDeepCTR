#-*- coding:utf-8 -*-
# @Time : 2019/10/4
# @Author : Botao Fan
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer

from DataPrepare import DataParser
from DeepFM.DeepFM import DeepFM
from DeepCrossNetwork.DeepCrossNetwork import DeepCrossNetwork
import config
import warnings
from eval_metrics import gini_norm
warnings.filterwarnings('ignore')


def _preprocess(df):
    cols = [c for c in df.columns if c not in ['id', 'target']]
    df['missing_feat'] = np.sum((df[cols] == -1).values, axis=1)
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    return df


def _load_train_data():
    data_path = config.DATA_PATH
    raw_data = pd.read_csv(data_path + 'train.csv')
    raw_data = _preprocess(raw_data)
    cols = [c for c in raw_data.columns if c not in ['id', 'target']]
    cols = [c for c in cols if c not in config.IGNORE_COLS]
    x_raw_df = raw_data[cols]
    y_raw_df = raw_data[['target']]
    train_index = int(raw_data.shape[0] * 0.75)
    data_parse = DataParser(x_raw_df, config.NUMERIC_COLS,
                            config.CATEGORICAL_COLS, config.IGNORE_COLS)
    df_val, df_idx, cate_dict = data_parse.parse()
    feature_size = df_idx.max().max() + 1
    field_size = df_val.shape[1]
    xv_train = df_val.iloc[:train_index, :].values
    xi_train = df_idx.iloc[:train_index, :].values
    y_train = y_raw_df.iloc[:train_index, :].values
    xv_test = df_val.iloc[train_index:, :].values
    xi_test = df_idx.iloc[train_index:, :].values
    y_test = y_raw_df.iloc[train_index:, :].values
    return xv_train, xi_train, y_train, xv_test, xi_test, y_test, feature_size, field_size, cate_dict


def _load_pred_data(cate_dict):
    data_path = config.DATA_PATH
    raw_data = pd.read_csv(data_path + 'test.csv')
    raw_data = _preprocess(raw_data)
    cols = [c for c in raw_data.columns if c not in ['id', 'target']]
    cols = [c for c in cols if c not in config.IGNORE_COLS]
    x_raw_df = raw_data[cols]
    data_parse = DataParser(x_raw_df, config.NUMERIC_COLS,
                            config.CATEGORICAL_COLS, config.IGNORE_COLS, cate_dict)
    test_val, test_idx, cate_dict = data_parse.parse()
    xv_test = test_val.values
    xi_test = test_idx.values
    return xv_test, xi_test


if __name__ == '__main__':
    xv_train, xi_train, y_train, xv_valid, xi_valid, y_valid, feature_size, field_size, cate_dict \
        = _load_train_data()
    xv_test, xi_test = _load_pred_data(cate_dict)
    #--- DeepFM Model---
    # dfm_params = {
    #     'feature_size': feature_size,
    #     'field_size': field_size,
    #     "use_fm": True,
    #     "use_deep": True,
    #     "embedding_size": 10,
    #     "dropout_fm": [1.0, 1.0],
    #     "deep_layers": [32, 32, 32],
    #     "dropout_deep": [0.5, 0.5, 0.5, 0.5],
    #     "deep_layers_activation": tf.nn.relu,
    #     "epoch": 20,
    #     "batch_size": 1024,
    #     "learning_rate": 0.001,
    #     "optimizer_type": "adam",
    #     "batch_norm": 1,
    #     "batch_norm_decay": 0.995,
    #     "l2_reg": 0.01,
    #     "eval_metric": gini_norm,
    #     "verbose": True
    # }
    # dfm = DeepFM(**dfm_params)
    # dfm.fit(xi_train, xv_train, y_train, xi_valid, xv_valid, y_valid)

    # --- Deep & Cross Network ---
    dcn_params = {
        'feature_size': feature_size,
        'field_size': field_size,
        "use_cross": True,
        "use_deep": True,
        "cross_layer_num": 2,
        "embedding_size": 8,
        "dropout_linear": [1.0],
        "deep_layers": [32, 32, 32],
        "dropout_deep": [0.5, 0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 2,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "eval_metric": gini_norm,
        "verbose": True
    }
    dcn = DeepCrossNetwork(**dcn_params)
    dcn.fit(xi_train, xv_train, y_train, xi_valid, xv_valid, y_valid)

    pred_df = pd.read_csv(config.DATA_PATH + 'test.csv')
    y_pred = dcn.predict(xi_test, xv_test)
    pred_df['target'] = y_pred
    pred_df[['id', 'target']].to_csv(config.DATA_PATH + 'dcn_submission.csv')




