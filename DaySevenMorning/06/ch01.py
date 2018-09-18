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
from sklearn.naive_bayes import   GaussianNB
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

#贝叶斯分类器
# 假设一个学校里有60%男生和40%女生.女生穿裤子的人数和穿裙子的人数相等,所有男生穿裤子.{一个人在远处随机看到了一个穿裤子的学生}.那么这个学生是女生的概率是多少?
# 使用贝叶斯定理,事件A是看到女生,事件B是看到一个穿裤子的学生.我们所要计算的是P(A|B).
# P(A)是忽略其它因素,看到女生的概率,在这里是40%
# P(A')是忽略其它因素,看到不是女生（即看到男生）的概率,在这里是60%
# P(B|A)是女生穿裤子的概率,在这里是50%
# P(B|A')是男生穿裤子的概率,在这里是100%
# P(B)是忽略其它因素,学生穿裤子的概率,P(B) = P(B|A)P(A) + P(B|A')P(A'),在这里是0.5×0.4 + 1×0.6 = 0.8.
# 根据贝叶斯定理,我们计算出后验概率P(A|B)
# P(A|B)=P(B|A)*P(A)/P(B)=0.25
def myGaussianNB():
    df.dropna(inplace=True)
    #cols = df_train.columns[:-1]
    #cols = ['年龄', '教育', '工龄', '地址', '收入', '信用卡负债']
    print('********************************************************')
    print('参考的特征工程字段如下')
    print('********************************************************')
    print(getFeature())
    cols = ['年龄','收入']
    #cols = ['年龄', '收入']
    Y = df['违约'].values

    X = df[cols].values
    #print(X)
    X = preprocessing.scale(X)#标准化，正太分布
    #print(print(X.mean(axis=0)))
    #print(X)

    # lr = LR()
    # lr.fit(X, Y)  # 训练模型
    num_folds =10
    seed =7
    kfold = KFold(n_splits=num_folds,random_state=seed)
    model = GaussianNB()
    model.fit(X,Y)
    scores = cross_val_score(model, X, Y,cv=kfold)

    print('模型准确率：%.2f%%' % (scores.mean() * 100))
    print('模型的平均正确率为:%s' % model.score(X, Y))  # 给出模型的平均正确率

if __name__ == "__main__":
    #getFeature()
    myGaussianNB()
    print("")