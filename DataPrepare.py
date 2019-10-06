#-*- coding:utf-8 -*-
# @Time : 2019/10/2
# @Author : Botao Fan
import pandas as pd


class DataParser(object):
    def __init__(self, df, num_cols=[], cate_cols=[], ignore_cols=[], cate_dict=None):
        cols_ = df.columns
        cols = []
        for col in cols_:
            if col not in ignore_cols:
                cols.append(col)
        self.df = df[cols]
        self.num_cols = num_cols
        self.cate_cols = cate_cols
        self.ignore_cols = ignore_cols
        self.cate_dict = self.generate_cate_dict() if cate_dict is None else cate_dict

    def generate_cate_dict(self):
        cate_dict = {}
        tc = 0
        for col in self.df.columns:
            if col in self.ignore_cols:
                continue
            elif col in self.num_cols:
                cate_dict[col] = tc
                tc += 1
            elif col in self.cate_cols:
                cates = self.df[col].unique()
                cates_count = len(cates)
                cate_dict[col] = dict(zip(cates, range(tc, tc + cates_count)))
                tc += cates_count
            else:
                raise KeyError('Column %s not in num_cols, cate_cols or ignore_cols' % col)
        return cate_dict

    def parse(self):
        df_val = self.df.copy()
        df_idx = self.df.copy()
        for col in self.df.columns:
            if col in self.ignore_cols:
                continue
            elif col in self.num_cols:
                df_val[col] = self.df[col]
                df_idx[col] = self.cate_dict[col]
            elif col in self.cate_cols:
                df_val[col] = 1
                df_idx[col] = self.df[col].map(self.cate_dict[col])
            else:
                raise KeyError('Column %s not in num_cols, cate_cols or ignore_cols' % col)
        return df_val, df_idx, self.cate_dict


if __name__=='__main__':
    df = pd.DataFrame([[1.1, 0, 1.3, 2, 1], [2.5, 1, 3.3, 3, 2], [4, 2, 3.3, 2, 3]],
                      columns=['n1', 'c1', 'n2', 'c2', 'ig1'])
    print df
    num_cols = ['n1', 'n2']
    cate_cols = ['c1', 'c2']
    ignore_cols = ['ig1']
    dp = DataParser(df, num_cols, cate_cols, ignore_cols)
    df_val, df_idx, cate_dict = dp.parse()
    print df_val
    print df_idx




