# -*- coding: utf-8 -*-
import pandas as pd
def read_csv():
    # 读csv文件
    data = pd.read_csv('bankloan.csv')
    print(data.describe())
    #print(data)

    # 返回前n行
    first_rows = data.head(n=5)
    #print(first_rows)

    #返回后n行
    last_rows = data.tail(n=5)
    #print(last_rows)

    # 返回全部列名
    cols = data.columns
    #print(cols)

    # 返回维度
    dimensision = data.shape
    #print(dimensision)

    # 返回所有数据 numpy格式
    #print(data.values)

    # 返回每一列数据类型
    #print(data.dtypes)

    # 返回指定行数据
    #print(data.loc[1])

    # 返回制定行制定类数据
    #print(data.loc[:,['违约']])

    # 去掉有缺失值的行
    print(data.dropna())

    # 对缺失值进行填充
    '''使用列的平均值进行填充'''
    print(data.fillna(data.mean()))
    '''0'''
    print(data.fillna(0))
    '''前向填充'''
    print(data.fillna(method="ffill"))

if __name__ == "__main__":
    read_csv()