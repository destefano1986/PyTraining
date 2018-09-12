# !/usr/bin/python
# coding: utf-8
# -*- coding: utf-8 -*-
import pandas as pd#科学分析
import numpy as np #科学计算
import matplotlib.pyplot as plt#图库
import os
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score

df = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "/data/bankloan.csv")


#提取特征工程
def getFeature():
    df.dropna(inplace=True)
    cols = df.columns[:-1]
    # cols = ['年龄', '教育', '工龄', '地址', '收入', '信用卡负债']
    #cols = ['年龄', '工龄', '信用卡负债']
    Y = df['违约'].values

    X = df[cols].values
    X = preprocessing.scale(X)
    # print(X)

    lr = LR()  # 建立随机逻辑回归模型，筛选变量
    lr.fit(X, Y)  # 训练模型

    #提取特征值
    coeflst = lr.coef_.tolist()[0]
    print(coeflst)
    coeflst2 = [True if i > 0.1  else False for i in coeflst]
    print(coeflst2)

    validfeatures = cols[np.where(coeflst2)]
    #print(validfeatures)

    #print('有效特征为:%s' % ','.join(validfeatures))

    return validfeatures

#使用逻辑回归模型做预测
#年龄,教育,工龄,地址,收入,负债率,信用卡负债,其他负债,违约
#[0.45066386518430646, 0.613154669967302, 0.230756206861915, 0.00743912269953533, 0.748472647142761, -0.1555494268746935, 0.3659428187410024, 0.3873206516024693]
#[True, True, True, False, True, False, True, True]
def mylr():
    df.dropna(inplace=True)
    #cols = df_train.columns[:-1]
    #cols = ['年龄', '教育', '工龄', '地址', '收入', '信用卡负债']
    print('********************************************************')
    print('参考的特征工程字段如下')
    print('********************************************************')
    print(getFeature())
    #cols = ['年龄','工龄','信用卡负债']
    cols = ['年龄', '收入']
    Y = df['违约'].values

    X = df[cols].values
    X = preprocessing.scale(X)#标准化，正太分布
    #print(X)

    lr = LR()
    lr.fit(X, Y)  # 训练模型

    scores = cross_val_score(lr, X, Y)

    print('模型准确率：%.2f%%' % (scores.mean() * 100))
    print('模型的平均正确率为:%s' % lr.score(X, Y))  # 给出模型的平均正确率

if __name__ == "__main__":
    #getFeature()
    mylr()
    print("")