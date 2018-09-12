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

#模型评估
def model_selection01():
    y_true = [3, -0.5, 2, 7]
    #y_pred = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    # tmp = pd.DataFrame(y_pred == y_true)
    # print(tmp)
    # tmp = tmp[0].value_counts().to_dict()
    # print(tmp[1])

    #均方误差（mean-square error, MSE）
    #是反映估计量与被估计量之间差异程度的一种度量
    #是各数据偏离真实值的距离平方和的平均数
    #该值越小越好
    print('MSE: %s' % mean_squared_error(y_true, y_pred))

    '''
    #回归分析（regression analysis)是确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法。
    #运用十分广泛，回归分析按照涉及的变量的多少，分为一元回归和多元回归分析；
    #按照因变量的多少，可分为简单回归分析和多重回归分析；
    #按照自变量和因变量之间的关系类型，可分为线性回归分析和非线性回归分析。
    #如果在回归分析中，只包括一个自变量和一个因变量，且二者的关系可用一条直线近似表示，这种回归分析称为一元线性回归分析。
    #如果回归分析中包括两个或两个以上的自变量，且自变量之间存在线性相关，则称为多重线性回归分析
    '''

    #线性回归决定系数R2
    #拟合优度
    #越大越好，自变量对因变量的解释程度越高，自变量引起的变动占总变动的百分比高。
    #观察点在回归直线附近越密集。
    #取值范围：0-1
    print('R2: %s' % r2_score(y_true, y_pred))

if __name__ == "__main__":
    model_selection01()
    print("")