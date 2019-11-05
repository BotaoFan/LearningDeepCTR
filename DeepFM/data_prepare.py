#-*- coding:utf-8 -*-
# @Time : 2019/11/1
# @Author : Botao Fan
import pandas as pd
from config import DATA_PATH, NUMERIC_COLUMN, CATEGORIAL_COLUMN, Y_COLUMN

class DataPrepare(object):
    def __init__(self, path=DATA_PATH, train_name='train.csv', test_name='test.csv',
                 num_col=NUMERIC_COLUMN, cat_col=CATEGORIAL_COLUMN, y_col=Y_COLUMN):
        self.path = path
        self.train_name = train_name
        self.test_name = test_name
        self.num_col = num_col
        self.cat_col = cat_col
        self.y_col = y_col
        self.feat_dict = None
        self.train, self.test = self._read_csv()

    def _read_csv(self):
        train = pd.read_csv(self.path + self.train_name)
        test = pd.read_csv(self.path + self.test_name)
        return train, test

    def _get_feat_dict(self):
        if self.feat_dict is None:
            self.feat_dict = dict()
            count = 0
            for col in self.num_col:
                self.feat_dict[col] = count
                count += 1
            for col in self.cat_col:
                items = self.train[col].unique()
                items_count = len(items)
                self.feat_dict[col] = {items[i]: count + i for i in range(items_count)}
                count += items_count
        else:
            return None

    def _get_idx_val(self, data):
        if self.feat_dict is None:
            self._get_feat_dict()

        idx = data.copy()
        val = data.copy()
        for col in self.num_col:
            idx[col] = self.feat_dict[col]
            val[col] = data[col]
        for col in self.cat_col:
            idx[col].apply(lambda x: self.feat_dict[col][x])
            val[col] = 1
        return idx.values, val.values

    def generate_train(self):
        train = self.train.copy()
        y = train[self.y_col].values
        train.drop(columns=self.y_col, inplace=True)
        self._get_feat_dict()
        idx, val = self._get_idx_val(train)
        return idx, val, y

    def generate_test(self):
        test = self.test.copy()
        self._get_feat_dict()
        idx, val = self._get_idx_val(test)
        return idx, val












