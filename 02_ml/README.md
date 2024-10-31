# 机器学习
> 平时注意构建知识体系，通过读论文和做实验不断为知识体系添砖加瓦。本章侧重理论与实践，系统设计参考[机器学习系统设计](../03_system/03_ml/README.md)


## 1. 面试要求

- 熟悉常见模型的**原理、代码、如何实际应用、优缺点、常见问题**等
  - 归纳偏置(Inductive Bias)，数据同分布(IID)

- 考察范围包括**ML breadth, ML depth, ML application, coding**
  - 可能持续被追问为什么? 为什么某个trick能起作用？
  - 算法背后的数学原理，写出其主要数学公式，并能进行白板推导
  - 一些较新的领域，会考察论文细节
  - 每一个算法的scale, 如何将算法map-reduce化
  - 每一个算法的复杂度、参数量、计算量


## 2. 八股问题实例
> 具体模型的细节与八股问题见本章节中具体模型页

- Generative vs Discriminative
  - A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. 
  - Discriminative models will generally outperform generative models on classification tasks. Discriminative model learns the predictive distribution p(y|x) directly while generative model learns the joint distribution p(x, y) then obtains the predictive distribution based on Bayes' rule.

- The bias-variance tradeoff is a central problem in supervised learning
  - Ideally, one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. 
  - High-variance learning methods may be able to represent their training set well but are at risk of overfitting to noisy or unrepresentative training data. 
  - In contrast, algorithms with high bias typically produce simpler models that don't tend to overfit but may underfit their training data, failing to capture important regularities.

- 怎么解决nn的 over-fitting
  - 数据角度，收集更多训练数据(more data)；求其次的话，数据增强(Data augmentation)；或者Pretrained model
  - 特征角度，Feature selection
  - 模型角度，降低模型复杂度，如神经网络的层数、宽度，树模型的树深度、剪枝；模型正则化(Regularization)，如正则约束L2；集成学习方法，bagging
  - 训练角度，Early stop

- 怎么解决under-fitting
  - 特征角度，增加新特征
  - 模型角度，增加模型复杂度，减少正则化系数
  - 训练角度，训练模型第一步就是要保证能够过拟合

- 怎么解决样本不平衡问题
  - [https://imbalanced-learn.org/en/stable/user_guide.html](https://imbalanced-learn.org/en/stable/user_guide.html)
  - 评价指标：AP(average_precision_score)
  - downsampling: faster convergence, save disk space, calibration(=upweight?)
  - upweight: every sample contribute the loss equality
  - long tail classification，只取头部80%的label，其他label mark as others
  - 极端imbalance，99.99% 和0.01%，outlier detection的方法
  - 样本多少可继续引申到样本的难易

- 怎么解决数据缺失的问题
  - [How to Handle Missing Data](https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4)

- 怎么解决类别变量中的高基数特征 high-cardinality

- 如何选择优化器
  - MSE, loglikelihood+GD
  - SGD-training data太大量
  - ADAM-sparse input

- 数据收集
  - production data, label
  - Internet dataset

- 分布不一致怎么解决
  - distribution不是特别指的feature的，也有label的。label只能说多收集data，还是balance data的问题。
  - data distribution 改变，就是做auto train, auto deploy.如果参数drop太多，只能人工干预重新训练

- 怎么提升模型的latency
  - 小模型
  - 知识蒸馏
  - squeeze model to 8bit or 4bit

- 模型的并行
  - 线性/逻辑回归
  - xgboost
  - cnn
  - RNN
  - transformer
  - 在深度学习框架中，单个张量的乘法内部会自动并行


## 3. 手写ML代码实例
> [ML code challenge](https://www.deep-ml.com/)

- 手写两层fully connected网络
- [手写CNN](./05_deep_learning.md)
- [手写KNN](./07_knn.md)
- [手写K-means](./09_k_means.md)
- 手写softmax的backpropagation
- 手写AUC
- 手写SGD
- 实现dropout，前向和后向
- 实现focal loss
- 手写multi head attention (MHA)
- 视觉：手写iou/nms
- NLP:
  - 手写n-gram
  - 手写tokenizer
    - [BPE tokenizer](https://colab.research.google.com/drive/1QLlQx_EjlZzBPsuj_ClrEDC0l8G-JuTn?usp=sharing#scrollTo=Nnjv2FLnX3rr)
    - [BPE tokenizer](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)
- follow-up问题
  - 给一个LSTM network的结构，计算how many parameters
  - convolution layer的output size怎么算? 写出公式
  - 设计一个sparse matrix (包括加减乘等运算)


## 参考
- [https://github.com/eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [https://github.com/resumejob/interview-questions](https://github.com/resumejob/interview-questions)
- [https://github.com/2019ChenGong/Machine-Learning-Notes](https://github.com/2019ChenGong/Machine-Learning-Notes)
- [https://github.com/ctgk/PRML](https://github.com/ctgk/PRML)
- [https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
- [https://docs.qq.com/doc/DR0ZBbmNKc0l3RGR2](https://docs.qq.com/doc/DR0ZBbmNKc0l3RGR2)
- [机器学习八股文的答案](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=998257&page=1&extra=)
- [ML, DL学习面试交流总结](https://www.1point3acres.com/bbs/thread-788612-1-1.html)
- [Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml)