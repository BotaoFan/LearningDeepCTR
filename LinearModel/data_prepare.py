#-*- coding:utf-8 -*-
# @Time : 2019/10/30
# @Author : Botao Fan
from config import DATA_PATH
import pandas as pd
import numpy as np

def data_prep(url=DATA_PATH, name='ua.base', user_dict=None, item_dict=None):
    col_name = ['user', 'item', 'rating', 'timestamp']
    data = pd.read_csv(url + name, delimiter='\t', names=col_name)
    y = np.reshape(data['rating'].apply(lambda x: 1 if x==1 else 0).values, [-1, 1])
    data.drop(columns=['rating', 'timestamp'], inplace=True)
    if user_dict is None:
        user_list = np.sort(data['user'].unique())
        item_list = np.sort(data['item'].unique())
        user_num = max(user_list)
        item_num = max(item_list)
        user_dict = {i + 1: i for i in range(user_num)}
        item_dict = {i + 1: i + user_num for i in range(item_num)}
    idx = data.copy()
    idx['user'] = data['user'].apply(lambda x: user_dict[x])
    idx['item'] = data['item'].apply(lambda x: item_dict[x])
    val = data.copy()
    val['user'] = 1
    val['item'] = 1
    idx, val = idx.values, val.values
    return idx, val, y, user_dict, item_dict


