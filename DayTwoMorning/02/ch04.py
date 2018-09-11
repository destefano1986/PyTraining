# !/usr/bin/python
# coding: utf-8
# -*- coding: utf-8 -*-
import pandas as pd#科学分析
import numpy as np #科学计算
import matplotlib.pyplot as plt#图库
import os
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 数据预处理，填充缺失值以及将特征中含有字符的转换为数值型
def pretreatment():
    data = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "/data/train.csv")
    # 将年龄这一列的数据缺失值进行填充
    data["Age"] = data["Age"].fillna(data["Age"].median())
    print(data.describe())  # 打印这一列特征中的特征值都有哪些

    print(data["Sex"].unique())
    '''
    将性别中的男女设置为0 1 值 
    机器学习不能处理的自字符值转换成能处理的数值
    loc定位到哪一行，将data['Sex'] == 'male'的样本Sex值改为0
    loc定位到哪一行，将data['Sex'] == 'female'的样本Sex值改为1
    '''
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    data["Embarked"] = data["Embarked"].fillna("S")  # 将登船地点同样转换成数值
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2
    #print(data["Embarked"].unique())
    #print(data)
    return data

#通过线性回归进行预测
def dome():
    data_train = pretreatment
    #data_test = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "/data/test.csv")
    cols = {'PassengerId': 1, 'Survived': 2, 'Pclass': 3, 'Name': 4, 'Sex': 5, 'Age': 6, 'SibSp': 7, 'Parch': 8,'Ticket': 9, 'Fare': 10, 'Cabin': 11, 'Embarked': 12}
    data_train.rename(columns=cols,inplace=True)
    #data_test.rename(columns=cols, inplace=True)
    #准备训练样本特征
    X_train = data_train[np.arange(10,11)]
    print(X_train.shape)
    #准备训练样本输出y
    Y_train = data_train[np.arange(2,3)]
    print(Y_train.shape)

    #scikit-learn的线性回归算法使用的是最小二乘法来实现
    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)

    #Survived = 0.37633435-0.03687421*SibSp+0.07020434*Parch
    print(regr.intercept_)
    print(regr.coef_)
    print('******************')

    # #准备测试样本特征
    # X_test = data_test[np.arange(7,9)]
    # print(X_test.shape)
    # # 准备测试样本输出y
    # Y_test = Y_train.as_matrix()
    # print(Y_test.shape)

    # 模型拟合测试集
    # Make predictions using the testing set
    Y_pred = regr.predict(X_train)
    print(len(Y_pred))
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print(mean_squared_error(Y_train, Y_pred))
    # Explained variance score: 1 is perfect prediction
    print(r2_score(Y_train, Y_pred))
    # Plot outputs
    print("+++++++++++++")
    print(X_train)
    print(Y_train)
    plt.scatter(X_train, Y_train, color='black')
    plt.plot(X_train, Y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

def dome1():
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

if __name__ == "__main__":
    print(pretreatment())
    dome()
    #dome1()
    print("")