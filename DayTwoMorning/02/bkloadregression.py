# -*- coding:utf-8 -*-
import pandas as pd#科学分析
import numpy as np #科学计算
import matplotlib.pyplot as plt#图库
import os
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report

#通过线性回归进行预测
def dome():
    data_train = pd.read_csv('bank_loan.csv', encoding='utf8')
    #print (data_train)
    X_train = data_train.loc[:12,['age','edu','workage','address','revenue','debtratio','creditdebt','otherdebt']]
    Y_train = data_train.loc[:12,['default']]

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    print (regr.score(X_train, Y_train))
    #print(regr.intercept_)
    print(regr.coef_)
    X_test = data_train.loc[13:,['age','edu','workage','address','revenue','debtratio','creditdebt','otherdebt']]
    Y_test = data_train.loc[13:,['default']]
    Y_test_pred = regr.predict(X_test)
    Y_test_pred[Y_test_pred > 0.5]=1
    Y_test_pred[Y_test_pred <= 0.5]=0
    print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_test_pred))
    print(classification_report(Y_test,Y_test_pred,target_names=['Default','Undefault']))

if __name__ == "__main__":
    dome()