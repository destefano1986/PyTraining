
# !/usr/bin/python
# coding: utf-8
# -*- coding: utf-8 -*-
import pandas as pd#科学分析
import numpy as np #科学计算
import matplotlib.pyplot as plt#图库
import os

#箱形图是一种用作显示一组数据分散情况资料的统计图
def quartile():
    # 读csv文件
    data = pd.read_csv('bankloan.csv', encoding = 'utf8')
    data = data.dropna()
    cols = {'年龄':1,'教育':2,'工龄':3,'地址':4,'收入':5,'负债率':6,'信用卡负债':7,'其他负债':8,'违约':9}
    data.rename(columns=cols,inplace=True)
    print(data.describe())
    # sym 调整好异常值的点的形状
    # whis 默认是1.5， 通过调整它的竖直来设置异常值显示的数量，
    # 如果想显示尽可能多的异常值，whis设置很小，否则很大
    plt.boxplot(data.as_matrix(columns=np.arange(1,10)), sym="o", whis=0.01)
    # plt.boxplot(data, sym ="o", whis = 0.01)
    # plt.boxplot(data, sym ="o", whis = 999)
    plt.show()

def quartile02():
    # 读csv文件
    data = pd.read_csv('train.csv', encoding = 'utf8')
    #print(data.info)
    #print(data.describe())
    data = set_Cabin_type(data)
    data = set_Age_default(data)
    print(data)
    # cols = {'PassengerId':1,'Survived':2,'Pclass':3,'Name':4,'Sex':5,'Age':6,'SibSp':7,'Parch':8,'Ticket':9,'Fare':10,'Cabin':11,'Embarked':12}
    # data.rename(columns=cols,inplace=True)
    plt.boxplot(data.as_matrix(columns=['Survived','Age','Pclass','SibSp','Parch','Fare']), sym="o", whis=1.5)
    plt.show()

#客舱预处理
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

#年龄预处理
def set_Age_default(df):
    m = df.loc[(df.Age.notnull()), 'Age'].mean()
    #print('m: %d' % m)
    df.loc[ (df.Age.isnull()), 'Age' ] = m
    return df

if __name__=="__main__":
    quartile()
    quartile02()

