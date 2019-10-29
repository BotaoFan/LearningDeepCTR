#-*- coding:utf-8 -*-
# @Time : 2019/10/28
# @Author : Botao Fan

from FM import FM
from config import DATA_PATH
from data_prepare import data_prep

if __name__ == '__main__':
    train_x, train_y, user_dict, item_dict = data_prep(url=DATA_PATH, name='ua.base', user_dict=None, item_dict=None)
    test_x, test_y, _, _ = data_prep(url=DATA_PATH, name='ua.test', user_dict=user_dict, item_dict=item_dict)
    num_param = train_x.shape[1]
    fm = FM(num_param, echo=200)
    fm.fit(train_x, train_y, test_x, test_y)