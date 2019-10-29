#-*- coding:utf-8 -*-
# @Time : 2019/10/28
# @Author : Botao Fan
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from config import DATA_PATH


def data_prep(url=DATA_PATH, name='ua.base', user_dict=None, item_dict=None):
    col_name = ['user', 'item', 'rating', 'timestamp']
    data = pd.read_csv(url + name, delimiter='\t', names=col_name)
    if 'rating' in data.columns:
        y = data[['rating']].values
        data.drop(columns=['timestamp', 'rating'], inplace=True)
    else:
        y = None
        data.drop(columns=['timestamp'], inplace=True)
    n, p = data.shape
    user_list = np.sort(data['user'].unique())
    item_list = np.sort(data['item'].unique())
    if user_dict is None:
        n_user = max(len(user_list), max(user_list))
        n_item = max(len(item_list), max(item_list))
        user_dict = {i + 1: i for i in range(n_user)}
        item_dict = {i + 1: i + n_user for i in range(n_item)}
    else:
        n_user = len(user_dict)
        n_item = len(item_dict)
    data_val = np.ones(n * p)
    data_row = range(n) + range(n)
    data_col = [user_dict[data['user'][i]] for i in range(n)] + [item_dict[data['item'][i]] for i in range(n)]
    x = csr_matrix((data_val, (data_row, data_col)), shape=(n, n_user + n_item)).toarray()
    return x, y, user_dict, item_dict
