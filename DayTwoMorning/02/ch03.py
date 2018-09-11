# !/usr/bin/python
# coding: utf-8
# -*- coding: utf-8 -*-
import pandas as pd#科学分析
import numpy as np #科学计算
import matplotlib.pyplot as plt#图库
import os

data = pd.read_csv('train.csv')
def aa():
    print(data.describe())
    print(data.info())


    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
    data.Survived.value_counts().plot(kind='bar')  # 柱状图
    plt.title(u"获救情况 (1为获救)")  # 标题
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 3), (0, 1))
    data.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"乘客等级分布")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data.Survived, data.Age)
    plt.ylabel(u"年龄")  # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布 (1为获救)")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    data.Age[data.Pclass == 1].plot(kind='kde')
    data.Age[data.Pclass == 2].plot(kind='kde')
    data.Age[data.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")  # plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')  # sets our legend for our graph.

    plt.subplot2grid((2, 3), (1, 2))
    data.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")

    plt.show()

#属性与获救结果的关联统计
#有图分析：乘客等级对获救的可能性有影响，等级为1的乘客，获救的概率高很多
def bb():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数
    Survived_0 = data.loc[data['年龄'] <= data['年龄'].mean, ['年龄']].value_counts()
    Survived_1 = data.loc[data['年龄'] > data['年龄'].mean, ['年龄']].value_counts()
    print (Survived_0, Survived_1)
    '''
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各乘客等级的获救情况")
    plt.xlabel(u"乘客等级")
    plt.ylabel(u"人数")
    plt.show()
    '''

#性别无疑也要作为重要特征加入最后的模型之中
def cc():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_m = data.Survived[data.Sex == 'male'].value_counts()
    Survived_f = data.Survived[data.Sex == 'female'].value_counts()
    df = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
    df.plot(kind='bar', stacked=True)
    plt.title(u"按性别看获救情况")
    plt.xlabel(u"性别")
    plt.ylabel(u"人数")

    plt.show()

#各种舱级别情况下各性别的获救情况
#女性高级仓获救人数较多ß
def dd():
    fig = plt.figure()
    fig.set(alpha=0.65)  # 设置图像透明度，无所谓
    plt.title(u"根据舱等级和性别的获救情况")

    ax1 = fig.add_subplot(141)
    data.Survived[data.Sex == 'female'][data.Pclass != 3].value_counts().plot(kind='bar',
                                                                                                label="female highclass",
                                                                                                color='#FA2479')
    ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
    ax1.legend([u"女性/高级舱"], loc='best')

    ax2 = fig.add_subplot(142, sharey=ax1)
    data.Survived[data.Sex == 'female'][data.Pclass == 3].value_counts().plot(kind='bar',
                                                                                                label='female, low class',
                                                                                                color='pink')
    ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"女性/低级舱"], loc='best')

    ax3 = fig.add_subplot(143, sharey=ax1)
    data.Survived[data.Sex == 'male'][data.Pclass != 3].value_counts().plot(kind='bar',
                                                                                              label='male, high class',
                                                                                              color='lightblue')
    ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/高级舱"], loc='best')

    ax4 = fig.add_subplot(144, sharey=ax1)
    data.Survived[data.Sex == 'male'][data.Pclass == 3].value_counts().plot(kind='bar',
                                                                                              label='male low class',
                                                                                              color='steelblue')
    ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/低级舱"], loc='best')

    plt.show()

#各登船港口的获救情况
def ee():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_0 = data.Embarked[data.Survived == 0].value_counts()
    Survived_1 = data.Embarked[data.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各登录港口乘客的获救情况")
    plt.xlabel(u"登录港口")
    plt.ylabel(u"人数")
    plt.show()

#堂兄弟/妹，孩子/父母有几人，对是否获救的影响
def ff():
    g = data.groupby(['SibSp', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)

    g = data.groupby(['Parch', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)

#有Cabin记录的获救概率稍高一些
def gg():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_cabin = data.Survived[pd.notnull(data.Cabin)].value_counts()
    Survived_nocabin = data.Survived[pd.isnull(data.Cabin)].value_counts()
    df = pd.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
    df.plot(kind='bar', stacked=True)
    plt.title(u"按Cabin有无看获救情况")
    plt.xlabel(u"Cabin有无")
    plt.ylabel(u"人数")
    plt.show()


if __name__ == "__main__":
    #aa()
    #bb()
    cc()
    #dd()
    ##ee()
    #ff()
    #ticket是船票编号，应该是unique的
    #gg()
    #print(data.Cabin.value_counts())
    print("")
