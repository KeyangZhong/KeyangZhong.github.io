---
title: 'Principal Component Analysis(PCA)'
date: 2023-09-06
permalink: /posts/2023/09/Principal Component Analysis(PCA)/
tags:
  - Machine Learning
  - Machine Learning Models
---
无监督模型--主成分分析PCA

参考：《动手学机器学习》； [主成分分析（PCA）原理详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/37777074)



## 数据降维
对高维复杂的数据，不同维度代表的特征可能存在关联，也可能存在无意义的噪声。因此，无论是监督学习还是无监督学习，我们都希望从中提取出具有代表性的，能最大限度保留数据本身信息的几个特征，从而实现降低数据维度，简化之后的分析和计算。\
降维具有如下一些优点：
- 使得数据集更易使用。
- 降低算法的计算开销。
- 去除噪声。
- 使得结果容易理解。

降维的算法有很多，比如奇异值分解(SVD)、主成分分析(PCA)、因子分析(FA)、独立成分分析(ICA)。
## PCA 概念
PCA是一种使用最广泛的数据降维算法。PCA的主要思想是将n维特征映射到k维上，这剩下的k维就是全新的正交特征，也称为主成分。PCA的工作就是从原始的空间中顺序地找一组相互正交的坐标轴。第一个新的坐标轴的选择是原始数据中方差最大的方向，第二个新的坐标轴选取时与第一个坐标轴正交的平面中使得方差最大的，第三个坐标轴是与第一，二个坐标轴正交的平面中方差最大的。以此类推，可以得到n个这样的坐标轴。通过这种方式获得新的坐标轴，但是但部分方差都包含在前k个坐标轴中，后面的坐标轴所函的方差几乎为0，则可以忽略余下的方差几乎为0的坐标，实现对数据特征的降维处理。
> 如何获得包含最大差异性的主成分方向？\
> 通过计算数据矩阵的协方差矩阵，然后得到协方差矩阵的特征值和特征向量，选择特征值最大的（即方差最大的）k个特征所对应的特征向量组成的矩阵。这样就可以实现将数据矩阵转换到新的空间当中，实现数据特征维度的降低。\
> 由于协方差矩阵的特征值和特征向量的计算有两种方法：特征值分解协方差矩阵，奇异值分解协方差矩阵，所以PCA算法有两种实现方法：**基于特征值分解的协方差矩阵实现PCA算法；基于SVD分解协方差矩阵实现PCA算法。**
## 方差，协方差
样本均值：
$$
\overline{x} = \frac{1}{n}\sum_{i=1}^Nx_i
$$
样本方差：
$$
S^2 = \frac{1}{n-1}\sum^{n}_{i=1}(x_i-\overline{x})^2
$$
样本X和样本Y的协方差：
$$
Cov(X,Y) = E[(X-E(X))[Y-E(Y)]]\\
=\frac{1}{n-1}\sum_{i=1}^n(x_i-\overline{x})(y_i-\overline{y})
$$
结论：
1. 方差的计算公式是针对一维特征，即针对同一特征不同样本的取值来进行计算得到；协方差则必须要求满足二维特征。
2. 方差和协方差的除以n-1，这是为了得到方差和协方差的无偏估计。

协方差为正时，X和Y是正相关关系；协方差为负时，X和Y是负相关关系；协方差为0时，说明X和Y是相互独立的。**Cov(X,X)就是X的方差**。当样本是n维度数据时，他们的协方差实际上是协方差矩阵。例如，对于3为数据（x,y,z），计算它的协方差就是：
$$
\text{Cov}(X,Y,Z) = 
\begin{bmatrix}
\text{Cov}(x, x) & \text{Cov}(x, y) & \text{Cov}(x, z) \\
\text{Cov}(y, x) & \text{Cov}(y, y) & \text{Cov}(y, z) \\
\text{Cov}(z, x) & \text{Cov}(z, y) & \text{Cov}(z, z) \\
\end{bmatrix}
$$
## 特征值分别矩阵原理
### （1）特征值和特征向量
如果一个向量v是矩阵A的特征向量，则一定可以表示成下面的形式：
$$
Av = \lambda v
$$
其中，$\lambda$是特征向量$v$对应的特征值，一个矩阵的一组特征向量是一组正交向量。
### （2）特征值分解矩阵
对于矩阵A，有一组特征向量v，将这组向量进行正交化单位化米酒能得到一组正交单位向量。特征值分解就是将矩阵A分解为如下式：
$$
A = Q\Sigma Q^{-1}
$$
其中$Q$是矩阵A的特征向量组成的矩阵，$\Sigma$是一个对角矩阵，对角线上的元素就是特征值。
## 基于特征值分解的协方差矩阵实现PCA算法
输入：数据集$X={x_1,x_2,x_3,...,x_n}$，需要降到k维。
1. 去平均值（即去中心化），每一个特征减去各自的平均值。
2. 计算协方差矩阵$\frac{1}{n}XX^T$（这里除或不除样本数量n或n-1对求出的特征值向量无影响）
3. 用特征值分解方法求协方差矩阵$\frac{1}{n}XX^T$的特征值和特征向量。
4. 对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量组成特征向量矩阵$P$。
5. 将数据转换到K个特征向量空间中，即$Y=PX$

更详细的参考，包括基于SVD分解协方差矩阵实现PCA算法：
1. [CodingLabs - PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html)；
2. [机器学习中SVD总结 (qq.com)](https://mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg)

##　Python代码实现
```python
##Python实现PCA
import numpy as np
def pca(X,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = X.shape
  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
  #normalization
  norm_X=X-mean
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  return data

X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

print(pca(X,1))
```

```python
##用sklearn的PCA
from sklearn.decomposition import PCA
import numpy as np
X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca=PCA(n_components=1)
pca.fit(X)
print(pca.transform(X))
```