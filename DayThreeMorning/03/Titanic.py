# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import warnings


def missing_value_rf(df):
    age_df = df[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[(df['Age'].notnull())]
    age_df_isnull = age_df.loc[(df['Age'].isnull())]
    X = age_df_notnull.values[:, 1:]
    Y = age_df_notnull.values[:, 0]
    RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    RFR.fit(X, Y)
    predictAges = RFR.predict(age_df_isnull.values[:, 1:])
    df.loc[df['Age'].isnull(), ['Age']] = predictAges
    return df

def factorizing(df):
    df['Embarked'] = pd.factorize(df['Embarked'])[0]
    df['Sex'] = pd.factorize(df['Sex'])[0]
    return df

def feature_standardization(df):
    #for i in ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']:
    for i in ['Age','Fare','SibSp','Parch']:
        df[i] = preprocessing.scale(df[i])
    return df

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_data = pd.read_csv('train.csv')  # 训练数据集
    train_data = train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]
    train_data = missing_value_rf(train_data)   #基于随机森林完成缺失值填充
    train_data.to_csv('train_data.csv',encoding='gbk')
    train_data.dropna(inplace=True)
    train_data = factorizing(train_data)
    print (train_data)
    train_data = feature_standardization(train_data)
    #train_data = shuffle(train_data)
    train_titanic = train_data.loc[:750,]
    test_titanic = train_data.loc[751:,]
    Y = train_titanic['Survived']
    X = train_titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    #X = train_titanic[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    lr = LR(max_iter=2000)  # 建立随机逻辑回归模型，筛选变量
    lr.fit(X, Y)  # 训练模型
    print (lr.score(X, Y))
    X_test = test_titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    #X = train_titanic[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    Y_test = test_titanic['Survived']
    Y_test_pred = lr.predict(X_test)
    print (X_test)
    #scores = cross_val_score(lr, X_test, Y_test)
    #print (scores)
    print(classification_report(Y_test,Y_test_pred,target_names=['Unsurvived','Survived']))
