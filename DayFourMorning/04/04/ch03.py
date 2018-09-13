'''
对于标称型数据来说，preprocessing.LabelBinarizer是一个很好用的工具。
比如可以把yes和no转化为0和1，或是把incident和normal转化为0和1。
当然，对于两类以上的标签也是适用的。这里举一个简单的例子，说明将标签二值化以及其逆过程
'''
# -*- coding: UTF-8 -*-
from sklearn import preprocessing
from sklearn import tree

# help(preprocessing.LabelBinarizer)#取消注释可以查看详细用法

#标签二值化
def lb():
    # 特征矩阵
    featureList=[[1,0],[1,1],[0,0],[0,1]]
    # 标签矩阵
    labelList=['yes', 'no', 'no', 'yes']
    # 将标签矩阵二值化
    lb = preprocessing.LabelBinarizer()
    dummY=lb.fit_transform(labelList)
    print(dummY)
    # 模型建立和训练
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(featureList, dummY)
    p=clf.predict([[0,1]])
    print(p)#取消注释可以查看p的值

    # 逆过程
    yesORno=lb.inverse_transform(p)
    print(yesORno)

'''
在机器学习中对于离散型的分类型的数据，需要对其进行数字化比如说性别这一属性，只能有男性或者女性或者其他这三种值，如何对这三个值进行数字化表达？
一种简单的方式就是男性为0，女性为1，其他为2，这样做有什么问题？
使用上面简单的序列对分类值进行表示后，进行模型训练时可能会产生一个问题就是特征的因为数字值得不同影响模型的训练效果，
在模型训练的过程中不同的值使得同一特征在样本中的权重可能发生变化，假如直接编码成1000，是不是比编码成1对模型的的影响更大。
为了解决上述的问题，使训练过程中不受到因为分类值表示的问题对模型产生的负面影响，引入独热码对分类型的特征进行独热码编码
'''
'''
假如只有一个特征是离散值：
{sex：{male， female，other}}
该特征总共有3个不同的分类值，此时需要3个bit位表示该特征是什么值，
对应bit位为1的位置对应原来的特征的值（一般情况下可以将原始的特征的取值进行排序，以便于后期使用），
此时得到独热码为{100}男性 ，{010}女性，{001}其他

假如多个特征需要独热码编码，那么久按照上面的方法依次将每个特征的独热码拼接起来：
{sex：{male， female，other}}
{grade：{一年级， 二年级，三年级， 四年级}}
此时对于输入为{sex：male； grade： 四年级}进行独热编码，
可以首先将sex按照上面的进行编码得到{100}，
然后按照grade进行编码为{0001}，
那么两者连接起来得到最后的码{1000001}；
'''
def onhot():
    from sklearn import preprocessing

    enc = preprocessing.OneHotEncoder()

    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])

    s = enc.transform([[0, 1, 3]]).toarray()

    print(s)
if __name__ == "__main__":
    #lb()
    onhot()