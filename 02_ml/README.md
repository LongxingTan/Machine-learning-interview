# 机器学习面试指南

> 面试准备需要技巧，但功夫在平时。建议：
> - 持续构建知识体系：阅读论文、动手实验
> - 面试前重点复习：机器学习**理论**和[模型代码实现](./99_ml_coding.md)
> - 系统设计相关内容请参考[3.3 机器学习系统设计](../03_system/03_ml/README.md)

## 1. 面试重点

### 1.1 核心要求
- 深入理解常见机器学习模型：
  - 原理推导
  - 代码实现
  - 实际应用场景
  - 优缺点分析
  - 常见问题及解决方案
- 掌握基础概念：
  - 归纳偏置(Inductive Bias)
  - 数据同分布(IID)

### 1.2 考察维度
- **ML广度**：对各类算法的理解
- **ML深度**：算法原理、数学推导
- **ML应用**：实际场景中的应用
- **Coding能力**：算法实现

### 1.3 面试技巧
- 白板推导：展示对算法原理的理解，而非死记硬背
- 持续追问：准备好回答"为什么"类问题
- 算法分析：掌握复杂度、参数量、计算量
- 扩展性：理解算法如何scale，如何map-reduce化
- 前沿技术：关注大模型等新领域，了解最新论文

## 2. 高频面试题

### 2.1 基础概念
- 生成模型 vs 判别模型
  - Discriminative model learns the predictive distribution **p(y|x)** directly.
  - Generative model learns the joint distribution **p(x, y)** then obtains the predictive distribution based on Bayes' rule.
  - A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data.
  - Discriminative models will generally outperform generative models on classification tasks.
- 偏差-方差权衡
  - how to track the tradeoff: Cross-Validation
  - Bias Variance Decomposition: Error = Bias \*\* 2 + Variance + Irreducible Error
  - Ideally, one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously.
  - High-variance learning methods may be able to represent their training set well but are at risk of overfitting to noisy or unrepresentative training data.
  - In contrast, algorithms with high bias typically produce simpler models that don't tend to overfit but may underfit their training data, failing to capture important regularities.
- 过拟合与
  - how to track: underfitting means large training error, large generalization error; overfitting means small training error, large generalization error
  - 数据角度: 收集更多训练数据；数据增强(Data augmentation)；或Pretrained model
  - 特征角度: Feature selection
  - 模型角度
    - 降低模型复杂度，如神经网络的层数、宽度，树模型的树深度、剪枝(pruning)；
    - 模型正则化(Regularization)，如正则约束L2，dropout
    - 集成学习方法，bagging
  - 训练角度: Early stop，weight decay
- 欠拟合
  - 特征角度: 增加新特征
  - 模型角度: 增加模型复杂度，减少正则化
  - 训练角度: 训练模型第一步就是要保证能够过拟合，增加epoch
- 样本不平衡
  - [https://imbalanced-learn.org/en/stable/user_guide.html](https://imbalanced-learn.org/en/stable/user_guide.html)
  - 评价指标：不要用准确率
  - down-sampling: faster convergence, save disk space, calibration. 样本多少可继续引申到样本的难易
  - up-weight: every sample contribute the loss equality
  - long tail classification，只取头部80%的label，其他label mark as others
  - 极端imbalance，99.99% 和0.01%，outlier detection的方法
- 数据缺失
  - [How to Handle Missing Data](https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4)
  - label data较少的情况: semi-supervised, few-shot
  - 特征列缺失：
    - 数据填充: mean, median, nan
    - 重要特征可通过额外建模进行预测
- 高基数特征处理
  - Feature Hashing
  - Target Encoding
  - Clustering Encoding
  - Embedding Encoding
- 梯度消失/爆炸
  - 梯度消失
    - stacking
    - 激活函数activations, 如ReLU: sigmoid只有靠近0的地方有梯度
    - LSTM (时间维度梯度)
    - Highway network
    - residual network (深度维梯度)
    - batch normalization
  - 梯度爆炸
    - gradient clipping
    - LSTM gate
- 分布不一致
  - distribution有feature和label的问题。label尽量多收集data，还是balance data的问题
  - data distribution 改变，就是做auto train, auto deploy. 如果性能drop太多，人工干预重新训练
  - 穿越特征也会造成分布不一致的表象，从避免穿越角度解决
- 线上线下不一致
  - model behaviors in production: data/feature distribution drift, feature bug
  - model generalization: offline metrics alignment
- 冷启动
  - 充分利用已有信息 (meta data)
  - 选择适合的模型 (two tower)
  - 流量调控

### 2.2 手写基础算法
- [KNN](./07_knn.md)
- [K-means](./09_k_means.md)
- SGD
- Softmax BackProp
- [MLP/CNN/LSTM](./05_deep_learning.md)
- Dropout实现
- Focal Loss
- 位置编码
- Multi-head Attention
- IOU/NMS
