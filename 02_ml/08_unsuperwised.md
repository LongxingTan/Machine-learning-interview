# 无监督学习

## 1. 统计学习
Clustering
- Centroid models: [k-means clustering](./09_k_means.md)
- Connectivity models: Hierarchical clustering
- Density models: DBSCAN

Gaussian Mixture Models
- EM是解KMS算法的方法，EM还可以解其他问题例如GMM

Latent semantic analysis

Hidden Markov Models (HMMs)
- Markov processes
- Transition probability and emission probability
- Viterbi algorithm

Dimension reduction techniques
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- T-sne


## 2. 深度学习自监督
contrastive learning 对比学习
> 相似的实例在投影空间中比较接近，不相似的实例在投影空间中距离比较远


## 3. 应用场景
实际应用中，无监督更加注重强特征的提取。

### 异常检测
见[异常检测专题](./16_anomaly.md)


### 风控
- 可解释性、攻防对抗
- 各颗粒度的唯一id，频率间隔等
- 风控中只有正样本和灰样本
- Positive-unlabeled learning


**风控场景特征**
  - 支付金额为整数的占比（刻画支付金额是不是都是整数）
  - 支付金额分布前10的占比（刻画支付金额是不是集中在几个数里）
  - 支付商铺的id占比（刻画支付金额是不是集中在几个店铺里）
  - 非运营时段夜间交易行为数量（高危支付行为数量）


**实践**
- 风控历史策略如何维护？
- 信贷风控建模时，正样本和负样本的区分可能比较模糊。可以结合业务识别是个体行为还是群体行为，如果是群体行为，做无监督聚类。先做常规的诈骗关键特征，通过聚集的关键特征与正常白样本有差异，做无监督。通过无监督制作标签转化为有监督分类。
- 所谓的业务知识，通过业务逻辑链条进行确认。比如诈骗电话归属地聚集，形成团伙。找到和团伙的通话记录对应的用户，形成较强的特征等。关联的聚集key是境外电话，通过异常聚集就能得到负样本。后面通过短信或app，转化为有监督


## reference
- [clustering](https://developers.google.com/machine-learning/clustering/clustering-algorithms)
