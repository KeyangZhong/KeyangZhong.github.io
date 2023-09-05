---
title: 'Logistic Regresssion'
date: 2023-09-05
permalink: /posts/2023/09/Logistic Regresssion/
tags:
  - Machine Learning
  - Machine Learning Models
---
简单逻辑斯蒂回归知识。

## 逻辑斯谛回归（Logistic Regression）
**线性回归处理回归问题；逻辑斯谛回归处理分类问题**

**Sigmoid function/Logistic function:**
$$
\sigma(t) = \frac{1}{1+\text{exp}(-t)}
$$
性质：

- Sigmoid函数是一个S型的函数，当自变量t趋近于正无穷，因变量趋近于1，当自变量趋近于负无穷是，因变量趋近于0；
- 使得任意实数映射到(0,1)区间，使其可用于将任意值函数转换为更适合二分类的函数。
- 使用该函数做分类时，不仅可以预测出类别，还能够得到近似概率预测，对很多需要概率辅助决策的任务很有用。
- 任意阶可导。

**为什么通过** **Sigmoid** **函数转换后得到的值，就可以认为是概率？**

因为 sigmoid 函数是伯努利分布的联系函数的反函数，它将线性函数映射到了伯努利分布的期望上，而伯努利分布的期望本身就是概率，因此，我们最终从逻辑斯蒂回归得到的输出，可以代表概率。

**分类问题的损失函数：Cross-Entropy**
$$
cost = -y*\log(\hat{y}) - (1-y)*\log(1-\hat{y})
$$

