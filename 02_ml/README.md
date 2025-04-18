# 机器学习

> 面试要有准备和技巧，但功夫在诗外。注意平时构建知识体系，读论文和做实验不断给体系添砖加瓦；面试前巩固机器学习**理论**和[代码](./99_ml_coding.md)
>
> - 本章侧重理论，系统设计参考[3.3 机器学习系统设计](../03_system/03_ml/README.md)

## 1. 面试要求

- 熟悉常见机器学习模型的**原理、代码、如何实际应用、优缺点、常见问题**

  - 归纳偏置(Inductive Bias)，数据同分布(IID)

- 考察范围如**ML breadth, ML depth, ML application, coding**。如果不知道答案不要乱编，承认不知道，并补充相关理解、做什么可以找到答案
  - 理解算法背后的原理，主要数学公式，并进行**白板推导介绍** (don’t memorize the formula but demonstrate understanding)
  - 可能被持续追问为什么? 某个trick为什么能起作用？
  - 每一个算法的**复杂度、参数量、计算量**
  - 每一个算法如何scale，如何将算法map-reduce化  
  - 较新的领域如[大模型](./12_llm.md)，会考察最新论文细节
  - 机器学习代码部分见 [ML coding collections](./99_ml_coding.md)

## 2. 八股问题实例

> 模型细节与具体问题见模型子页面。以下实例回答注意如何安框架分条陈述

- Generative vs Discriminative

  - Discriminative model learns the predictive distribution **p(y|x)** directly.
  - Generative model learns the joint distribution **p(x, y)** then obtains the predictive distribution based on Bayes' rule.
  - A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data.
  - Discriminative models will generally outperform generative models on classification tasks.

- The bias-variance tradeoff

  - how to track the tradeoff: Cross-Validation
  - Bias Variance Decomposition: Error = Bias \*\* 2 + Variance + Irreducible Error
  - Ideally, one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously.
  - High-variance learning methods may be able to represent their training set well but are at risk of overfitting to noisy or unrepresentative training data.
  - In contrast, algorithms with high bias typically produce simpler models that don't tend to overfit but may underfit their training data, failing to capture important regularities.

- 怎么解决over-fitting

  - how to track: underfitting means large training error, large generalization error; overfitting means small training error, large generalization error
  - 数据角度: 收集更多训练数据；数据增强(Data augmentation)；或Pretrained model
  - 特征角度: Feature selection
  - 模型角度
    - 降低模型复杂度，如神经网络的层数、宽度，树模型的树深度、剪枝(pruning)；
    - 模型正则化(Regularization)，如正则约束L2，dropout
    - 集成学习方法，bagging
  - 训练角度: Early stop，weight decay

- 怎么解决under-fitting

  - 特征角度: 增加新特征
  - 模型角度: 增加模型复杂度，减少正则化
  - 训练角度: 训练模型第一步就是要保证能够过拟合，增加epoch

- 怎么解决样本不平衡问题

  - [https://imbalanced-learn.org/en/stable/user_guide.html](https://imbalanced-learn.org/en/stable/user_guide.html)
  - 评价指标：不要用准确率
  - down-sampling: faster convergence, save disk space, calibration. 样本多少可继续引申到样本的难易
  - up-weight: every sample contribute the loss equality
  - long tail classification，只取头部80%的label，其他label mark as others
  - 极端imbalance，99.99% 和0.01%，outlier detection的方法

- 怎么解决数据缺失的问题

  - [How to Handle Missing Data](https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4)
  - label data较少的情况: semi-supervised, few-shot
  - 特征列缺失：
    - 数据填充: mean, median, nan
    - 重要特征可通过额外建模进行预测

- 怎么解决类别变量中的高基数(high-cardinality)特征

  - Feature Hashing
  - Target Encoding
  - Clustering Encoding
  - Embedding Encoding

- 如何选择优化器

  - MSE, loglikelihood+GD
  - SGD-training data太大
  - ADAM-sparse input

- 怎么解决Gradient Vanishing & Exploding

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

- 怎么解决分布不一致

  - distribution有feature和label的问题。label尽量多收集data，还是balance data的问题
  - data distribution 改变，就是做auto train, auto deploy. 如果性能drop太多，人工干预重新训练
  - 穿越特征也会造成分布不一致的表象，从避免穿越角度解决

- 怎么解决线上线下不一致

  - model behaviors in production: data/feature distribution drift, feature bug
  - model generalization: offline metrics alignment

- curse of dimensionality

  - Feature Selection
  - PCA
  - embedding

- 怎么提升模型latency

  - 小模型或剪枝(pruning)
  - 知识蒸馏
  - squeeze model to 8bit or 4bit

- 模型的并行

  - 线性/逻辑回归
  - xgboost
  - cnn
  - RNN
  - transformer
  - 在深度学习框架中，单个张量的乘法内部会自动并行

- 冷启动

  - 充分利用已有信息 (meta data)
  - 选择适合的模型 (two tower)
  - 流量调控

- Out-of-vocabulary

  - unknown

## 3. 手写ML代码实例

> - [ML code collections](./99_ml_coding.md)

- [手写KNN](./07_knn.md)
- [手写K-means](./09_k_means.md)
- [手写AUC](./01_metrics.md)
- 手写SGD
- 手写softmax的backpropagation
- [手写两层MLP](./05_deep_learning.md)
- [手写CNN](./05_deep_learning.md)
  - convolution layer的output size怎么算? 写出公式
- 实现dropout，前向和后向
- 实现focal loss
- 手写LSTM
  - 给定LSTM结构，计算参数量
- NLP:
  - 手写n-gram
  - 手写tokenizer
    - [BPE tokenizer](https://colab.research.google.com/drive/1QLlQx_EjlZzBPsuj_ClrEDC0l8G-JuTn?usp=sharing#scrollTo=Nnjv2FLnX3rr)
    - [BPE tokenizer](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)
  - 白板介绍位置编码
  - 手写multi head attention (MHA)
- 视觉:
  - 手写iou/nms

## 参考

- [https://github.com/2019ChenGong/Machine-Learning-Notes](https://github.com/2019ChenGong/Machine-Learning-Notes)
- [https://github.com/ctgk/PRML](https://github.com/ctgk/PRML)
- [https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
- [https://docs.qq.com/doc/DR0ZBbmNKc0l3RGR2](https://docs.qq.com/doc/DR0ZBbmNKc0l3RGR2)
- [机器学习八股文的答案](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=998257&page=1&extra=)
- [ML, DL学习面试交流总结](https://www.1point3acres.com/bbs/thread-788612-1-1.html)
- [Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [https://github.com/bitterengsci/algorithm](https://github.com/bitterengsci/algorithm/blob/master/royal%20algorithm/Machine%20Leanrning.md)
- [Pros and cons of various Machine Learning algorithms](https://towardsdatascience.com/pros-and-cons-of-various-classification-ml-algorithms-3b5bfb3c87d6)
- https://defiant-show-3ca.notion.site/Deep-learning-specialization-b69a42ecb14446f39bd93fd0f15965d5
