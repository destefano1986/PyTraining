import numpy as np
import numpy.random as random
#生成函数
def generate():
    '''3行2列二维数组'''
    a = np.array([[1,2],[3,4],[5,6]])
    print(a)

    b = np.array([[1,2],[3,4],[5,6]],dtype=float)
    print(b)

    c = np.asarray(b,dtype=int,order='C')
    print(c)

    '''全为0的数组'''
    a0 = np.zeros(6,dtype=float,order='C')
    print(a0)
    a01 = np.zeros([3,2],dtype=float,order='C')
    print(a01)
    a02 = np.zeros_like(a0,dtype=float,order='C')
    print(a02)

    '''全为1的数组'''
    a1 = np.ones(6,dtype=float,order='C')
    print(a1)
    a11 = np.ones([3,2],dtype=float,order='C')
    print(a11)
    a12 = np.ones_like(a1,dtype=float,order='C')
    print(a12)

    '''范围数组'''
    a2 = np.arange(1,7,1,dtype=float)
    print(a2)
    print(a2.reshape(3,2))

    '''空数组'''
    a3 = np.empty(6,dtype=float,order='C')
    print(a3)
    a31 = np.empty([3,2],dtype=float,order='F')
    print(a31)

    '''N*N的单位矩阵'''
    a4 = np.eye(3)
    print(a4)
    a41 = np.identity(3)
    print(a41)

    '''常数填充的数组'''
    a5 = np.full((3,2),6)
    print(a5)
#矩阵函数
def matrix():
    '''将一维数组转换为方阵（非对角线元素为0)'''
    a = np.diag([1,2,3])
    print(a)

    '''以一维数组的形式返回方阵的对角线'''
    b = np.diag(np.full(3,6))
    print(b)

    '''矩阵乘法'''
    c = np.arange(6).reshape(3,2)
    print(c)
    d = np.arange(4).reshape(2,2)
    print(d)
    e = np.dot(c,d)
    print(e)

    '''计算对角线元素的和'''
    f = np.trace(np.eye(5))
    print(f)
#排序函数
def sort():
    '''默认升序'''
    a = np.sort([9,2,3,4,5,6,1])
    b = np.sort(a)
    print(b)

    data = np.array([
        [1,2,1],
        [0,3,1],
        [2,1,4],
        [1,3,1]
    ])
    '''默认axis=-1行升序，axis=0列升序'''
    c = np.sort(data,axis=0)
    print(c)

    '''返回ndarray中的元素，排除重复的元素之后'''
    d = np.unique(data)
    print(d)
    print("*************")
    data01 = np.array([
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 4],
        [1, 3, 1]
    ])
    data02 = np.array([
        [2, 3, 2],
        [1, 4, 2],
        [3, 2, 5],
        [2, 4, 2]
    ])
    '''并集'''
    e = np.union1d(data01, data02)
    print(e)
    '''交集'''
    f = np.intersect1d(data01,data02)
    print(f)
    '''差集'''
    g = np.setdiff1d(data01,data02)
    print(g)
    '''对称差'''
    h = np.setxor1d(data01,data02)
    print(h)

#一元计算函数
def onefunc():
    '''绝对值'''
    print(np.abs([-1.2, 1.2]))
    '''平均值'''
    print(np.mean(np.eye(3)))
    '''x^0.5'''
    print(np.sqrt(np.arange(6).reshape(3,2)))
    '''x^2'''
    print(np.square(np.arange(6).reshape(3,2)))
    '''e^x'''
    print(np.exp(1))
    '''log log10 log2 log1p'''
    print(np.log10(100))
    print(np.log2(8))
    print(np.log(2.718281828459045))
    print(np.log1p(1.718281828459045))#log1p(x) := log(1+x)
    '''正负号，1正 0 -1负'''
    print(np.sign([-2,0,2]))
    '''向上取整'''
    print(np.ceil((-2.11,0,2.11)))
    '''向下取整'''
    print(np.floor((-2.11, 0, 2.11)))
    '''四舍五入到最近的整数，保留dtype'''
    print(np.rint((-2.11, 0, 2.11,2.56)))
    '''数组的小数和整数部分以两个独立的数组方式返回'''
    print(np.modf((-2.11, 0, 2.11)))
    '''判断是否是NaN的bool型数组'''
    print(np.isnan((-2.11, 0, 2.11)))
    '''判断是否是有穷的bool型数组'''
    print(np.isfinite((-2.11, 0, 2.11)))
    '''判断是否是无穷的bool型数组'''
    print(np.isinf((-2.11, 0, 2.11)))
    '''普通型和双曲型三角函数'''
    data = np.array([[1,2,3],[4,5,6]])
    print(np.sin(data))#对矩阵data中每个元素取正弦,sin(x)
    print(np.cos(data))#对矩阵data中每个元素取余弦,cos(x)
    print(np.sinh(data))
    print(np.cosh(data))
    print(np.tan(data))#对矩阵data中每个元素取正切,tan(x)
    print(np.tanh(data))
    '''反三角函数和双曲型反三角函数'''
    '''计算各元素not x的真值，相当于-ndarray'''
    print(np.logical_not(data))
    print(np.logical_and(data,data))
#多元函数
def morefunc():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    '''相加'''
    print(np.add(data,data))
    '''相减'''
    print(np.subtract(data, data))
    '''相乘'''
    print(np.multiply(data, data))
    '''相除'''
    print(np.divide(data, data))
    '''圆整除法（丢弃余数）'''
    print(np.floor_divide(data, data))
    '''次方'''
    print(np.power(data, data))
    '''求模'''
    print(np.mod(data, data))
    '''求最大值,比最大值小的用大值替代'''
    print(np.maximum(data, data))
    '''求最小值,比最小值小的用小值替代'''
    print(np.minimum(data, data))
    '''将参数2中的符号赋予参数1'''
    print(np.copysign(data,[-1]))
    '''>'''
    print(np.greater(data,np.arange(6).reshape(2,3)))
    '''>='''
    print(np.greater_equal(data, np.arange(6).reshape(2, 3)))
    '''<'''
    print(np.less(data,np.arange(6).reshape(2,3)))
    '''<='''
    print(np.less_equal(data, np.arange(6).reshape(2, 3)))
    '''=='''
    print(np.equal(data,data))
    '''!='''
    print(np.not_equal(data,data))
    '''&'''
    print(np.logical_and(True, False))
    '''|'''
    print(np.logical_or(True, False))
    '''^'''
    print(np.logical_xor(True, False))
    '''计算两个ndarray的矩阵内积'''
    print(np.dot(data,np.arange(6).reshape(3,2)))
    '''生成一个索引器'''
    print(np.ix_(np.arange(6)))

#文件读写
def io():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    '''将ndarray保存到文件名为[string].npy的文件中（无压缩）'''
    np.save('data',data)
    '''将所有的ndarray压缩保存到文件名为[string].npy的文件中'''
    np.savez('dataz',data,np.arange(6).reshape(3,2))
    '''将ndarray写入文件，格式为fmt'''
    np.savetxt('data.txt',(data),fmt='%s %s %s',newline='\n')
    '''读取文件名string的文件内容并转化为ndarray对象'''
    print(np.load('data.npy'))
    print(np.load('dataz.npz')["arr_0"])
    print(np.loadtxt('data.txt','\n'))

#随机生成
def ran():
    # 设置随机数种子
    random.seed(100)

    # 产生一个1x3，[0,1)之间的浮点型随机数
    # array([[ 0.37454012,  0.95071431,  0.73199394]])
    print(random.rand(1, 3))

    # 产生一个[0,1)之间的浮点型随机数
    print(random.random())

    # 指定大小产生[0,1)之间的浮点型随机数array，
    print(random.random((3, 3)))
    print(random.sample((3, 3)))

    # 产生10个[1,6)之间的浮点型随机数
    print(5 * random.random(10) + 1)
    print(random.uniform(1, 6, 10))

    # 产生10个[1,6]之间的整型随机数
    print(random.randint(1, 6, 10))

    # 产生2x5的标准正态分布样本
    print(random.normal(size=(5, 2)))

    # 产生5个，n=5，p=0.5的二项分布样本,
    # 在概率论和统计学里面，带有参数n和p的二项分布表示的是n次独立试验的成功次数的概率分布
    print(random.binomial(n=5, p=0.5, size=10))

    a = np.arange(10)

    # 从a中有回放的随机采样7个
    print(random.choice(a, 7))

    # 从a中无回放的随机采样7个
    print(random.choice(a, 7, replace=False))

    # 对a进行乱序并返回一个新的array
    b = random.permutation(a)
    print(b)

    # 对a进行in-place乱序
    random.shuffle(a)
    print(a)

    # 生成一个长度为9的随机bytes序列并作为str返回
    # '\x96\x9d\xd1?\xe6\x18\xbb\x9a\xec'
    print(random.bytes(9))

#练习题
'''使用循环和向量化两种不同的方法来计算 100 以内的质数之和'''
#判断质数
def checkprime(x):
    if x<=1:
        return False;
    prime=True;
    for i in range(2 , int(1+x/2)):
        if x%i == 0:
            prime = False;
            break;
    return prime;

def sum(n=100):
    sum=0
    for i in range(1, n+1):
        if( True == checkprime(i)):
            sum += i
    return sum

def sum_numpy(n=100):
    a = np.arange(1,n+1)
    ## 此处代码用到了 np.vectorize，可以把外置函数应用到向量的每个元素
    check_prime_vec = np.vectorize(checkprime)
    return np.sum(a[check_prime_vec(a)])



'''
违反直觉的概率题例子：一个选手去参加一个TV秀，有三扇门，其中一扇门后有奖品，这扇门只有主持人知道。
选手先随机选一扇门，但并不打开，主持人看到后，会打开其余两扇门中没有奖品的一扇门。
然后，主持人问选手，是否要改变一开始的选择？

这个问题的答案是应该改变一开始的选择。在第一次选择的时候，选错的概率是2/3，选对的概率是1/3。
第一次选择之后，主持人相当于帮忙剔除了一个错误答案，
所以如果一开始选的是错的，这时候换掉就选对了；
而如果一开始就选对，则这时候换掉就错了。
根据以上，一开始选错的概率就是换掉之后选对的概率（2/3），这个概率大于一开始就选对的概率（1/3），所以应该换
'''
def test():
    import numpy.random as random

    random.seed(42)

    # 做10000次实验
    n_tests = 10000

    # 生成每次实验的奖品所在的门的编号
    # 0表示第一扇门，1表示第二扇门，2表示第三扇门
    winning_doors = random.randint(0, 3, n_tests)

    # 记录如果换门的中奖次数
    change_mind_wins = 0

    # 记录如果坚持的中奖次数
    insist_wins = 0

    # winning_door就是获胜门的编号
    for winning_door in winning_doors:

        # 随机挑了一扇门
        first_try = random.randint(0, 3)

        # 其他门的编号
        remaining_choices = [i for i in range(3) if i != first_try]

        # 没有奖品的门的编号，这个信息只有主持人知道
        wrong_choices = [i for i in range(3) if i != winning_door]

        # 一开始选择的门主持人没法打开，所以从主持人可以打开的门中剔除
        if first_try in wrong_choices:
            wrong_choices.remove(first_try)

        # 这时wrong_choices变量就是主持人可以打开的门的编号
        # 注意此时如果一开始选择正确，则可以打开的门是两扇，主持人随便开一扇门
        # 如果一开始选到了空门，则主持人只能打开剩下一扇空门
        screened_out = random.choice(wrong_choices)
        remaining_choices.remove(screened_out)

        # 所以虽然代码写了好些行，如果策略固定的话，
        # 改变主意的获胜概率就是一开始选错的概率，是2/3
        # 而坚持选择的获胜概率就是一开始就选对的概率，是1/3

        # 现在除了一开始选择的编号，和主持人帮助剔除的错误编号，只剩下一扇门
        # 如果要改变注意则这扇门就是最终的选择
        changed_mind_try = remaining_choices[0]

        # 结果揭晓，记录下来
        change_mind_wins += 1 if changed_mind_try == winning_door else 0
        insist_wins += 1 if first_try == winning_door else 0

    # 输出10000次测试的最终结果，和推导的结果差不多：
    # You win 6616 out of 10000 tests if you changed your mind
    # You win 3384 out of 10000 tests if you insist on the initial choice
    print(
        'You win {1} out of {0} tests if you changed your mind\n'
        'You win {2} out of {0} tests if you insist on the initial choice'.format(
            n_tests, change_mind_wins, insist_wins
        )
    )
if __name__ == "__main__":
    #generate()
    #matrix()
    #sort()
    #onefunc()
    #morefunc()
    #io()
    #print(sum())
    print(sum_numpy())
    #ran()
    #test()
    #print()