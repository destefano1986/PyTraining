# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def bankloan(para):
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数
    Survived_0 = data.default[data[para] <= np.mean(data[para])].value_counts()
    Survived_1 = data.default[data[para] > np.mean(data[para])].value_counts()
    #Survived_0 = data.default[data['age'] <= np.mean(data['age'])].value_counts()
    #Survived_1 = data.default[data['age'] > np.mean(data['age'])].value_counts()
    df = pd.DataFrame({'below avg': Survived_1, 'above avg': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.xlabel(str(para))
    plt.ylabel("count")
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('bank_loan.csv', encoding='utf8')
    data.dropna(inplace=True)
    #print(data)
    lst=['age','edu','workage','address','revenue','debtratio','creditdebt','otherdebt']
    for i in lst:
        bankloan(i)
    #print (data.describe())