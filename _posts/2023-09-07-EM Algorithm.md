---
title: 'EM Algorithm'
date: 2023-09-07
permalink: /posts/2023/09/EM Algorithm/
tags:
  - Machine Learning
  - Machine Learning Models
  - Unsupervised Learning
---

EM算法是一种迭代算法，用于求解含有隐变量的概率模型参数的极大似然估计。

参考：
1. 《动手学机器学习》
2. [EM（最大期望）算法推导、GMM的应用与代码实现 - 颀周 - 博客园 (cnblogs.com)](https://www.cnblogs.com/qizhou/p/13100817.html)
3. [EM算法详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/40991784)

## 概述
EM算法是一种迭代优化策略，由于它的计算方法中每一次迭代都分为两步，其中一个为期望步（E步），另一个为极大步（M步），所以算法被称为EM算法（Expectation-Maximization）。EM算法的基本思想是：根据已经给出的观测数据，估计出模型参数的值；然后根据上一步估计的参数值估计缺失数据的值，再根据估计出的缺失的数据加上之前已经观测到的数据重新再对参数值进行估计，反复迭代，直至最后收敛，迭代结束。
## 预备知识
### 极大似然估计
