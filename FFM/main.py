#-*- coding:utf-8 -*-
# @Time : 2019/10/30
# @Author : Botao Fan

from config import DATA_PATH
from FFM import FFM
from data_prepare import data_prep

if __name__ == '__main__':
    train_idx, train_val, train_y, user_dict, item_dict = data_prep(DATA_PATH, 'ua.base')
    test_idx, test_val, test_y, _, _ = data_prep(DATA_PATH, 'ua.test', user_dict, item_dict)
    ffm = FFM(param_size=train_idx.shape[1])
    ffm.fit(train_idx, train_val, train_y, test_idx, test_val, test_y)
