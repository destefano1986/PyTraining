import numpy as np
import pandas as pd

def f(str):
    x = len(str)
    if x < 5:
        return True
    else:
        return False

def f1(x):
    if x < 5:
        return True
    else:
        return False

#Series函数
def func():
    '''生成一个Series'''
    obj = pd.Series(range(4), index=['d', 'b', 'a', 'c'])
    print(obj)

    '''Series重建序号'''
    print(obj.reindex(['a', 'b', 'c', 'd', 'e']))
    print(pd.Series({'a':1,'b':2}))

    '''
    method=’ffill’或’pad 前向填充
    method=’bfill’或’backfill 后向填充
    '''
    se01=pd.Series(['blue','red','black'],index=[0,2,4])
    print(se01)
    se02=se01.reindex(range(6),method='ffill')
    print(se02)

    '''丢弃指定项'''
    print(se02.drop(5))

    '''字典映射'''
    print(se02.map(f))

#排序函数
def func_sort():
    '''生成序列'''
    se01 = pd.Series(np.arange(6))
    print(se01)
    '''序列按照降序排列'''
    se02 = se01.sort_index(ascending=False)
    print(se02)
    '''按降序进行排名'''
    se03 = se01.rank(ascending=False, method='max')
    print(se03)

#DataFrame常用函数
def func_df():
    '''字典构建DataFrame'''
    dict = {'a':np.arange(3),'b':np.arange(3,6),'c':np.arange(6,9)}
    frame = pd.DataFrame(dict)
    print(frame)
    '''数组构建DataFrame'''
    print(pd.DataFrame(np.arange(9).reshape(3,3),columns=('one','two','three'),index=(1,2,3)).rank(axis=1))
    '''Series构建DataFrame'''
    print(pd.DataFrame(pd.Series(np.arange(9),dtype=float).values.reshape(3,3),columns=('one','two','three'),index=('A','B','C')))

    '''按指定的值对DataFrame进行排序'''
    print(frame.sort_values(by='b',ascending=False))

    '''重建索引'''
    a = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'b', 'c'], columns=['a', 'b', 'c'])
    print(a.reindex(index=['a', 'b', 'c', 'd'], method='ffill', columns=['one', 'two', 'three','four']))

    '''丢弃指定轴上的指定项'''
    print(frame.drop(index=0,columns=['a','b']))

#DataFrame汇总统计函数
def func_df_statistical():
    df = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['one', 'two', 'three'], columns=['a', 'b', 'c'])
    print(df)
    print(df.count())
    print(df.count(axis=1))
    '''
    汇总统计：
    计数
    平均值
    标准差
    最小值
    第一个四分位数（First Quartile）Q1表示，公式为1+(n-1)*0.25=1.5
    中位数Q2，公式为1+(n-1)*0.5 = 3.0
    第三个四分位数（Third Quartile）Q3表示，公式为1+(n-1)*0.75 = 
    最大值
    '''
    print(df.describe())
    '''每列最大值'''
    print(df.max())
    '''每列最小值'''
    print(df.min())
    '''返回含有最大值的index的Series'''
    print(df.idxmax())
    '''返回含有最小值的index的Series'''
    print(df.idxmin())
    '''计算样本的分位数'''
    print(df.quantile())
    '''每列求和'''
    print(df.sum())
    '''每列求均值'''
    print(df.mean())
    '''每列求中位数'''
    print(df.median())
    '''每列平均绝对离差'''
    print(df.mad())
    '''每列方差'''
    print(df.var())
    '''每列标准差'''
    print(df.std())
    '''每列偏度'''
    print(df.skew())

def func_sort_cal():
    df_temp = pd.DataFrame(np.arange(1,10).reshape((3, 3)), index=['one', 'two', 'three'], columns=['a', 'b', 'c'])
    df = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['one', 'two', 'three'], columns=['a', 'b', 'c'])
    print(df)
    '''更加索引排序'''
    print(df.sort_values(by=['a','b','c'],ascending=False))
    '''元素级相加，对齐时找不到元素默认用fill_value'''
    print(df.add(df_temp,fill_value=0))
    '''元素级相减，对齐时找不到元素默认用fill_value'''
    print(df.sub(df_temp, fill_value=0))
    '''元素级相乘，对齐时找不到元素默认用fill_value'''
    print(df.mul(df_temp, fill_value=0))
    '''元素级相除法，对齐时找不到元素默认用fill_value'''
    print(df.div(df_temp, fill_value=0))
    '''形成新的一维数组'''
    print(df.apply(f))
    '''应用到各个元素上'''
    print(df.applymap(f1))
    '''累加'''
    print(df.cumsum())
if __name__ == "__main__":
    #func()
    #func_sort()
    #func_df()
    #func_df_statistical()
    #func_sort_cal()
    print()