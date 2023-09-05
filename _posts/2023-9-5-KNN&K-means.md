参考链接：\
[机器学习中 K近邻法(knn)与k-means的区别 - Niucas_MAO - 博客园 (cnblogs.com)](https://www.cnblogs.com/PiPifamily/p/8520405.html)\
[机器学习实战教程（一）：K-近邻（KNN）算法（史诗级干货长文） (cuijiahua.com)](https://cuijiahua.com/blog/2017/11/ml_1_knn.html)
K近邻法（knn）是一种基本的分类与回归方法。k-means是一种简单而有效的聚类方法。
## KNN（K-Nearest Neighbor）
算法思路：\
如果一个样本在特征空间中的K个最相似（即特征空间中最邻近）的样本中的大多数属于某个类别，则该样本也属于这个类别。\
KNN算法步骤：
1. 计算一直类别数据集中的点与当前点之间的距离；
2. 按照距离递增次序排序；
3. 选取与当前点距离最小的K个点；
4. 确定前K个点所在类别出现的频率；
5. 返回前K个点所出现频率最高的类别作为当前点的预测分类。


KNN的三个基本要素：
1. k值的选择：k值的选择会对结果产生重大影响。较小的k值可以减少近似误差，但是会增加估计误差；较大的k值可以减小估计误差，但是会增加近似误差。一般而言，通常采用交叉验证法来选取最优的k值。
2. 距离度量：距离反映了特征空间中两个实例的相似程度。可以采用欧氏距离、曼哈顿距离等。
3. 分类决策规则：往往采用多数表决。


**优点**
- 简单好用，容易理解，精度高，理论成熟，既可以用来做分类也可以用来做回归；
- 可用于数值型数据和离散型数据；
- 训练时间复杂度为O(n)；无数据输入假定；
- 对异常值不敏感

**缺点**
- 计算复杂性高；空间复杂性高；
- 样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
- 一般数值很大的时候不用这个，计算量太大。但是单个样本又不能太少，否则容易发生误分。
- 最大的缺点是无法给出数据的内在含义。


**代码实现：**
```python
import numpy as np
import operator
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX,dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    # np.tile将inX扩展成dataSetSize*1的矩阵，然后与dataSet相减
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet 
    # 计算inX与训练集各点的欧式距离
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5

    # argsort函数返回的是数组值从小到大的索引值，即inX与训练集各点的欧式距离从小到大的索引值
    sortedDistIndicies = distance.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] # 获取标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 # 统计标签出现次数
        
    # 根据字典的值进行降序排序，sorted返回的是一个列表
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    # 返回出现次数最多的标签
    return sortedClassCount[0][0]
if __name__ == '__main__':
    group, labels = createDataSet()
    test = [0,0] # 测试数据
    test_class = classify0(test, group, labels, 3)
    print(test_class)
```

## K-means
算法步骤：
1. 从n个数据中随机选择K个对象作为初始聚类中心；
2. 计算每个样本与聚类中心点的距离，将样本划分到最近的中心点；
3. 计算划分到每个类别中的所有样本特征的均值，并将该均值作为每个类新的聚类中心；
4. 循环步骤2和3，直到每个聚类簇不再发生变化为止，输出最终的聚类中心以及每个样本所属的类别。

K-means方法的基本要素：

1. K值的选择：即类别的选择，与K近邻中K值的确定方法类似
2. 距离度量，可以采用欧式距离，曼哈顿距离等。

优点：
简单，快速，适合常规数据集
劣势：

- K值难确定
- 复杂度与样本呈线性关系,O(k\*m\*n)
- 很难发现任意形状的簇

**代码实现：**
```python
def euclidean_distance(x, y):
    return np.linalg.norm(x - y,ord=2) # 二范数，即欧氏距离

def random_centroids(X, k):
    # 随机选取k个中心点
    n_samples, n_features = np.shape(X)
    centroids = np.zeros((k, n_features)) 
    centroids = X[np.random.choice(range(n_samples),k)]
    return centroids

def kmeans(X, k, max_iter=1000):
    centroids = random_centroids(X, k) # 随机选取k个中心点
    for _ in range(max_iter):
        # 计算每个样本到中心点的距离
        distances = np.array([np.linalg.norm(x - centroids,ord=2,axis=1) for x in X])
        # 选取距离最近的中心点
        labels = np.argmin(distances, axis=1)
        # 更新中心点
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids, labels

def show_cluster(X, k, centrodis):
    # 绘制聚类结果
    n_samples, n_features = np.shape(X)
    distances = np.array([np.linalg.norm(x - centrodis,ord=2,axis=1) for x in X])
    labels = np.argmin(distances, axis=1)
    for i in range(k):
        plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1])
    plt.scatter(centrodis[:, 0], centrodis[:, 1], marker='x', s=50)
    plt.show()

if __name__ == '__main__':
    X = np.random.randn(100, 2)
    centrodis, labels = kmeans(X, 2)
    show_cluster(X, 2, centrodis)
```
