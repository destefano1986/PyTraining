import matplotlib.pyplot as plt
import numpy as np

def boxplot():
#箱形图boxplot
    np.random.seed(100)

    # 生成一组随机数，数量为1000
    data = np.random.normal(size=(1000,), loc=0, scale=1)

    # sym 调整好异常值的点的形状
    # whis 默认是1.5， 通过调整它的竖直来设置异常值显示的数量，
    # 如果想显示尽可能多的异常值，whis设置很小，否则很大
    #plt.boxplot(data)
    plt.boxplot(data, sym="o", whis=1.5)
    # plt.boxplot(data, sym ="o", whis = 0.01)
    # plt.boxplot(data, sym ="o", whis = 999)
    plt.show()


#pyplot饼图的绘制
def pie():
    labels = '武汉','上海','深圳','广州'
    sizes = [15,30,45,10]
    '''将30在饼图中凸显出来，凸显的比例为0.1'''
    explode = (0,0.1,0,0)
    '''
    autopct 数据显示方式   shadow是否带有阴影效果    startangle 饼图起始角度
    '''
    plt.pie(sizes,explode=explode,labels=labels,autopct='%0.1f%%',shadow=True,startangle=90)
    #plt.axis('equal')   #正圆形饼图
    plt.show()

#pyplot直方图的绘制
def hist():
    np.random.seed(0)
    mu, signs = 100, 20  # 均值和标准差
    data = np.random.normal(mu, signs, size=100)
    '''
    hist的常用参数六个，第一个是必须的，后面可选
    arr: 需要计算直方图的一维数组
    bins: 直方图的柱数，可选项，默认为10
    normed: 1表示纵坐标为数据频率，0表示数据出现的频数热点。默认为0
    facecolor: 直方图颜色
    edgecolor: 直方图边框颜色
    alpha: 透明度
    histtype: 直方图类型，‘bar', ‘barstacked', ‘step', ‘stepfilled'
    '''
    plt.hist(data, bins=3, normed=0, histtype='stepfilled',edgecolor=None ,facecolor='red', alpha=0.9);
    plt.title('Histogram');
    plt.show();

#条形图
def bar():
    # 第一步，取出一张白纸
    fig = plt.figure(1)

    # 第二步，确定绘图范围，由于只需要画一张图，所以我们将整张白纸作为绘图的范围
    ax1 = plt.subplot(111)

    # 第三步，整理我们准备绘制的数据
    data = np.array([15, 20, 18, 25])

    # 第四步，准备绘制条形图，思考绘制条形图需要确定那些要素
    # 1、绘制的条形宽度
    # 2、绘制的条形位置(中心)
    # 3、条形图的高度（数据值）
    width = 0.5
    x_bar = np.arange(4)

    # 第五步，绘制条形图的主体，条形图实质上就是一系列的矩形元素，我们通过plt.bar函数来绘制条形图
    rect = ax1.bar(left=x_bar, height=data, width=width, color="lightblue")

    # 第六步，向各条形上添加数据标签
    for rec in rect:
        x = rec.get_x()
        height = rec.get_height()
        ax1.text(x + 0.1, 1.02 * height, str(height))

    # 第七步，绘制x，y坐标轴刻度及标签，标题
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(("first", "second", "third", "fourth"))
    ax1.set_ylabel("sales")
    ax1.set_title("The Sales in 2016")
    ax1.grid(True)
    ax1.set_ylim(0, 28)
    plt.show()

#pyplot散点图的绘制
def scatter():
    # 产生测试数据
    x = np.arange(1, 10)
    y = x
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Scatter Plot')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    lValue = x
    ax1.scatter(x, y, c='r', s=100, linewidths=lValue, marker='o')
    # 设置图标
    plt.legend('x1')
    # 显示所画的图
    plt.show()

#pyplot散点图的绘制
def scatter02():
    n = 100
    x = np.random.normal(0, 1, n)  # 平均值为0，方差为1，生成100个数
    y = np.random.normal(0, 1, n)
    t = np.arctan2(x, y)  # for color value，对应cmap

    plt.scatter(x, y, s=75, c=t, alpha=0.5)  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    #boxplot()
    #pie()
    #hist()
    bar()
    #scatter()
    #scatter02()
    #print("")