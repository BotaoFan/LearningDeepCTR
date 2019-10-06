#-*- coding:utf-8 -*-
# @Time : 2019/10/6
# @Author : Botao Fan

import numpy as np


def gini(actual, pred):
    assert len(actual) == len(pred)
    all = np.asarray(np.c_[actual, pred, range(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    total_loss = all[:, 0].sum()
    gini = all[:, 0].cumsum().sum() / total_loss
    gini -= (len(actual) + 1) * 0.5
    gini = gini / len(actual)
    return gini


def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)