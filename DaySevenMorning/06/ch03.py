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
from sklearn.model_selection import KFold
from sklearn.naive_bayes import  GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
df = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "/data/bankloan.csv")


#提取特征工程
def getFeature():
    df.dropna(inplace=True)
    cols = df.columns[:-1]
    #cols = ['年龄', '教育', '工龄', '地址', '收入', '信用卡负债']
    #cols = ['年龄', '工龄', '信用卡负债']
    Y = df['违约'].values

    X = df[cols].values
    #X = preprocessing.scale(X)
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


def mySVM():
    df.dropna(inplace=True)
    #cols = df_train.columns[:-1]
    #cols = ['年龄', '教育', '工龄', '地址', '收入', '信用卡负债']
    #cols = ['年龄',  '工龄', '收入', '信用卡负债']
    print('********************************************************')
    print('参考的特征工程字段如下')
    print('********************************************************')
    print(getFeature())
    cols = ['年龄','收入']
    #cols = ['年龄', '收入']
    Y = df['违约'].values
    X = df[cols].values
    X = preprocessing.scale(X)#标准化，正太分布

    num_folds =10
    seed =7
    kfold = KFold(n_splits=num_folds,random_state=seed)
    model = SVC()
    model.fit(X,Y) # 训练模型
    scores = cross_val_score(model, X, Y,cv=kfold)

    print('模型准确率：%.2f%%' % (scores.mean() * 100))
    print('模型的平均正确率为:%s' % model.score(X, Y))  # 给出模型的平均正确率

if __name__ == "__main__":
    #getFeature()
    mySVM()
    print("")