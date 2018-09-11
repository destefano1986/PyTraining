import  pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
def programmer_1():
    # 参数初始化
    filename = r'bankloan.xls'
    data = pd.read_excel(filename)
    x = data.iloc[:, :8].as_matrix()  # 使用pandas读取文件  就可以不用管label column标签
    y = data.iloc[:, 8].as_matrix()

    rlr = RLR()  # 建立随机逻辑回归模型，进行特征选择和变量筛选
    rlr.fit(x, y)  # 训练模型
    egeList = rlr.get_support()  # 获取筛选后的特征
    egeList = np.append(egeList, False)  # 往numpy数组中 添加一个False元素  使用np.append(array,ele)方法
    print("rlr.get_support():")
    print(egeList)
    print(u'随机逻辑回归模型特征选择结束！！！')
    print(u'有效特征为：%s' % ','.join(data.columns[egeList]))
    x = data[data.columns[egeList]].as_matrix()  # 筛选好特征值

    lr = LR()  # 建立逻辑回归模型
    lr.fit(x, y)  # 用筛选后的特征进行训练
    print(u'逻辑回归训练模型结束！！！')
    print(u'模型的平均正确率：%s' % lr.score(x, y))  # 给出模型的平均正确率，本例为81.4%

if __name__ =="__main__":
    programmer_1()