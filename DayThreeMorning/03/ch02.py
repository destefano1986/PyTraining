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

#标准化(中心化与缩放)，也称去均值和方差按比例缩放
def feature_standardization_01():
    X_train = np.array([[1., -1., 2.],[2., 0., 0.],[0., 1., -1.]])
    X_scaled = preprocessing.scale(X_train)
    print(X_scaled)
    #经过缩放后的数据具有零均值以及标准方差:
    print(X_scaled.mean(axis=0))#[0. 0. 0.]
    print(X_scaled.std(axis=0))#[1. 1. 1.]

#将数据矩阵缩放到``[0, 1]``
#是将特征缩放到给定的最小值和最大值之间，通常在0和1之间，
#也可以将每个特征的最大绝对值转换至单位大小，分别使用 MinMaxScaler 和 MaxAbsScaler 实现。
#使用这种缩放的目的包括实现特征极小方差的鲁棒性以及在稀疏矩阵中保留零元素
def feature_standardization_02():
    X_train = np.array([[1., -1., 2.],[2., 0., 0.],[0., 1., -1.]])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    print(X_train_minmax)
    #检查缩放器（scaler）属性，来观察在训练集中学习到的转换操作的基本性质:
    print(min_max_scaler.scale_)#[0.5        0.5        0.33333333]
    print(min_max_scaler.scale_)#[0.5        0.5        0.33333333]

#非线性转换
#类似于缩放， QuantileTransformer 类将每个特征缩放在同样的范围或分布情况下。
#通过执行一个秩转换能够使异常的分布平滑化，并且能够比缩放更少地受到离群值的影响
def feature_standardization_nonlinear_03():
    quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    X_train = np.array([0, 25, 50, 75, 100]).reshape(1,-1)
    X_train_trans = quantile_transformer.fit_transform(X_train)
    print(X_train_trans)
    print("")

#归一化
#归一化是缩放单个样本以具有单位范数的过程。
#如果你计划使用二次形式(如点积或任何其他核函数)来量化任何样本间的相似度，则此过程将非常有用。
#这个观点基于向量空间模型(Vector Space Model),经常在文本分类和内容聚类中使用
#函数normalize提供了一个快速简单的方法在类似数组的数据集上执行操作，使用 l1 或 l2 范式:
def feature_normalize():
    X = [[1., -1., 2.],[2., 0., 0.],[0., 1., -1.]]
    X_normalized = preprocessing.normalize(X, norm='l2')
    print(X_normalized)
    print(X_normalized.mean(axis=0))
    print("")

if __name__ == "__main__":
    #feature_standardization_01()
    #feature_standardization_02()
    #feature_standardization_nonlinear_03()
    #feature_normalize()
    print("")