# 机器学习
> 面试需要一些准备和技巧，但功夫在诗外。平时注意构建知识体系，论文和实验不断给体系添砖加瓦。本章侧重理论部分，系统设计参考[3.3 机器学习系统设计](../03_system/03_ml/README.md)


## 1. 面试要求

- 熟悉常见模型的**原理、代码、如何实际应用、优缺点、常见问题**等
  - 归纳偏置(Inductive Bias)，数据同分布(IID)

- 考察范围包括**ML breadth, ML depth, ML application, coding**  
  - 算法背后的数学原理，写出主要数学公式，并能进行**白板推导介绍**
  - 一些较新的领域如[大模型](./12_llm.md)，会考察论文细节
  - 可能被持续追问为什么? 某个trick为什么能起作用？
  - 每一个算法如何scale，如何将算法map-reduce化
  - 每一个算法的**复杂度、参数量、计算量**

- [简历中介绍自己的机器学习项目](./22_project.md)


## 2. 八股问题实例
> 模型细节与八股见具体模型页面

- Generative vs Discriminative
  - A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. 
  - Discriminative models will generally outperform generative models on classification tasks. Discriminative model learns the predictive distribution p(y|x) directly while generative model learns the joint distribution p(x, y) then obtains the predictive distribution based on Bayes' rule.

- The bias-variance tradeoff
  - Bias Variance Decomposition: Error = Bias ** 2 + Variance + Irreducible Error
  - Ideally, one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. 
  - High-variance learning methods may be able to represent their training set well but are at risk of overfitting to noisy or unrepresentative training data. 
  - In contrast, algorithms with high bias typically produce simpler models that don't tend to overfit but may underfit their training data, failing to capture important regularities.

- 怎么解决over-fitting
  - track: underfitting means large training error, large generalization error; overfitting means small training error, large generalization error
  - 数据角度，收集更多训练数据(more data)；求其次，数据增强(Data augmentation)；或Pretrained model
  - 特征角度，Feature selection
  - 模型角度
    - 降低模型复杂度，如神经网络的层数、宽度，树模型的树深度、剪枝；
    - 模型正则化(Regularization)，如正则约束L2，dropout
    - 集成学习方法，bagging
  - 训练角度，Early stop，weight decay

- 怎么解决under-fitting
  - 特征角度，增加新特征
  - 模型角度，增加模型复杂度，减少正则化系数
  - 训练角度，训练模型第一步就是要保证能够过拟合，增加epoch

- 怎么解决样本不平衡问题
  - [https://imbalanced-learn.org/en/stable/user_guide.html](https://imbalanced-learn.org/en/stable/user_guide.html)
  - 评价指标：AP(average_precision_score)
  - downsampling: faster convergence, save disk space, calibration. 样本多少可继续引申到样本的难易
  - upweight: every sample contribute the loss equality
  - long tail classification，只取头部80%的label，其他label mark as others
  - 极端imbalance，99.99% 和0.01%，outlier detection的方法

- 怎么解决数据缺失的问题
  - [How to Handle Missing Data](https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4)
  - label data较少的情况

- 怎么解决类别变量中的高基数特征 high-cardinality
  - Feature Hashing
  - Target Encoding
  - Clustering Encoding
  - Embedding Encoding

- 如何选择优化器
  - MSE, loglikelihood+GD
  - SGD-training data太大量
  - ADAM-sparse input

- 怎么解决Gradient Vanishing & Exploding
  - 梯度消失
    - 激活函数activations, 如ReLU
    - residual network
    - batch normalization
  - 梯度爆炸
    - gradient clipping
    - LSTM gate

- 数据收集
  - production data, label
  - Internet dataset

- 分布不一致怎么解决
  - distribution有feature和label的问题。label尽量多收集data，还是balance data的问题
  - data distribution 改变，就是做auto train, auto deploy. 如果性能drop太多，人工干预重新训练
  - 穿越特征也会造成分布不一致的表象，从避免穿越角度解决

- 线上线下不一致
  - model behaviors in production: data/feature distribution drift, feature bug
  - model generalization: offline metrics alignment

- curse of dimensionality
  - Feature Selection
  - PCA
  - embedding

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

- [手写KNN](./07_knn.md)  
- [手写K-means](./09_k_means.md)
- 手写softmax的backpropagation
- 手写AUC
- 手写SGD
- 手写两层fully connected网络
- [手写CNN](./05_deep_learning.md)
  - convolution layer的output size怎么算? 写出公式
- 实现dropout，前向和后向
- 实现focal loss
- 手写LSTM
  - 给一个LSTM network的结构，计算参数量
- NLP:
  - 手写n-gram
  - 手写tokenizer
    - [BPE tokenizer](https://colab.research.google.com/drive/1QLlQx_EjlZzBPsuj_ClrEDC0l8G-JuTn?usp=sharing#scrollTo=Nnjv2FLnX3rr)
    - [BPE tokenizer](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)
  - 白板介绍位置编码
  - 手写multi head attention (MHA)
- 视觉：
  - 手写iou/nms


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
- [https://github.com/bitterengsci/algorithm](https://github.com/bitterengsci/algorithm/blob/master/royal%20algorithm/Machine%20Leanrning.md)
- [Pros and cons of various Machine Learning algorithms](https://towardsdatascience.com/pros-and-cons-of-various-classification-ml-algorithms-3b5bfb3c87d6)
- [10min pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [60min pytorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
