#-*- coding:utf-8 -*-
# @Time : 2019/10/28
# @Author : Botao Fan
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from config import DATA_PATH


def data_prep(url=DATA_PATH, name='ua.base'):
    col_name = ['user', 'item', 'rating', 'timestamp']
    data = pd.read_csv(url + name, delimiter='\t', names=col_name)
    if 'rating' in data.columns:
        y = data['rating'].values
        data.drop(columns=['timestamp', 'rating'], inplace=True)
    else:
        y = None
        data.drop(columns=['timestamp'], inplace=True)
    n, p = data.shape
    user_list = np.sort(data['user'].unique())
    item_list = np.sort(data['item'].unique())
    n_user = len(user_list)
    n_item = len(item_list)
    user_dict = {user_list[i]: i for i in range(n_user)}
    item_dict = {item_list[i]: i + n_user for i in range(n_item)}
    data_val = np.ones(n * p)
    data_x = range(n) + range(n)
    data_y = [user_dict[data['user'][i]] for i in range(n)] + [item_dict[data['item'][i]] for i in range(n)]
    x = csr_matrix((data_val, (data_x, data_y)), shape=(n, n_user + n_item)).toarray()
    return x, y, [user_dict, item_dict]
