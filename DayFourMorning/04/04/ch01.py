from time import time
# !/usr/bin/python
# coding: utf-8
# -*- coding: utf-8 -*-
import pandas as pd#科学分析
import numpy as np #科学计算
import matplotlib.pyplot as plt#图库
import os
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.externals import joblib

df = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "/data/bankloan.csv")
df.dropna(inplace=True)
cols = ['年龄', '收入']
#cols = df.columns[:-1]
Y = df['违约'].values
# 标准正态化数据
X = scale(df[cols].values)
row, column = X.shape
# 去重得到的标签数量为[0 1]
n_digits = len(np.unique(Y))
# 每个点的分类标签
labels = Y

#k_means算法
def bench_k_means(estimator, name, data):

    print(82 * '_')
    '''
    模型效果评估
    # inertia,样本距离最近的聚类中心的距离总和
    # ARI兰德系数（Adjusted rand index）,RI取值范围为[0,1]，值越大意味着聚类结果与真实情况越吻合
    # AMI调整互信息（Adjusted Mutual Information）,AMI取值范围[-1,1]，都是值越大说明聚类效果越好
    # Silhouette轮廓系数,适用于实际类别信息未知的情况，用来计算所有样本的平均轮廓系数
    # v-meas得分
    # compl完整性得分
    # homo同质化得分
    '''
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
    t0 = time()
    # 持久化k_means模型
    joblib.dump(estimator, 'k_means.pk')
    #加载k_means模型
    estimator = joblib.load('k_means.pk')

    #得到训练结果
    y_pred = estimator.fit_predict(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=row)))
    y_pred = estimator.fit_predict(data)
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("银行违约聚类")
    plt.show()

if __name__ == "__main__":
    '''
    KMeans参数说明
    #n_clusters缺省值=8, 生成的聚类数，即产生的质心（centroids）数
    #n_init缺省值=10, 用不同的质心初始化值运行算法的次数，最终解是在inertia意义下选出的最优结果
    #init：有三个可选值：'k-means++'， 'random'，或者传递一个ndarray向量。此参数指定初始化方法，默认值为'k-means++'
    #1 ‘k-means++’ 用一种特殊的方法选定初始质心从而能加速迭代过程的收敛
    #2 ‘random’ 随机从训练数据中选取初始质心。
    #3 如果传递的是一个ndarray，则应该形如 (n_clusters, n_features) 并给出初始质心。
    #precompute_distances：三个可选值，‘auto’，True或者False。预计算距离，计算速度更快但占用更多内存。   
    '''
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),name="k-means++", data=X)
    print("")
