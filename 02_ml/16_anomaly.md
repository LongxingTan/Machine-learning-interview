# 异常检测与风控

- 异常类型：Point anomaly, Contextual anomaly, Collective anomaly
- 具体类型：根据取值范围、常数波动、固定斜率波动、滑动聚合加方差、一阶差分、局部异常
- 任务类型：有监督，无监督；连续异常检测，离散异常检测
- 模型类型：距离，分类

## 1. 模型

**规则、经验**

- remove outlier: Random sample consensus (RANSAC) is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates

**3sigma**

- Z-score标准化，数据满足正态分布

**PCA**

- 降维

**Local Outlier Factor**

**孤立森林**

**Robust Random Cut Forest**

**One-Class SVM**

**DBSCAN**

**autoencoder**

**GAN**

### 2 风控

- 信贷风控(join新数据)、电商风控(埋点 event tracking)、内容风控(正则规则)
- 可解释性、攻防对抗
- 风控中只有正样本和灰样本
- Positive-unlabeled learning
- 各颗粒度的唯一id，频率间隔等

**风控场景特征**

- 支付金额为整数的占比（刻画支付金额是不是都是整数）
- 支付金额分布前10的占比（刻画支付金额是不是集中在几个数里）
- 支付商铺的id占比（刻画支付金额是不是集中在几个店铺里）
- 非运营时段夜间交易行为数量（高危支付行为数量）

**实践**

- 风控历史策略如何维护？
- 信贷风控建模时，正样本和负样本的区分可能比较模糊。可以结合业务识别是个体行为还是群体行为，如果是群体行为，做无监督聚类。先做常规的诈骗关键特征，通过聚集的关键特征与正常白样本有差异，做无监督。通过无监督制作标签转化为有监督分类。
- 通过**业务**逻辑链条进行确认，类似图的社区发现。比如通过诈骗电话归属地聚集，cluster形成团伙；找到和团伙的通话记录对应的用户，形成较强的特征等。关联的聚集key是境外电话，通过异常聚集就能得到负样本；而后通过短信或app，转化为有监督

**模型校准Calibration**

- Isotonic Regression: 模型的原始输出和真实标签作为输入，使用 Isotonic Regression 拟合得到一个单调递增的函数
- Temperature Scaling: 深度学习的softmax
- `from sklearn.calibration import calibration_curve, CalibratedClassifierCV`
- [On Calibration of Modern Neural Networks](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf)

**指标**

- psi、iv、ks、auc、lift

## reference

- [Warden: Real Time Anomaly Detection at Pinterest](https://medium.com/pinterest-engineering/warden-real-time-anomaly-detection-at-pinterest-210c122f6afa)
- [Tubi 时间序列 KPI 的异常值检测 - Tubi 中国团队的文章 - 知乎](https://zhuanlan.zhihu.com/p/642174241)
- [基于AI算法的数据库异常监测系统的设计与实现](https://tech.meituan.com/2022/09/01/database-monitoring-based-on-ai.html)
- [网易如何做到数据指标异常发现和诊断分析？](https://mp.weixin.qq.com/s/wr9XvBNRBeKfp6acxkXc2A)
- [How our content abuse defense systems work to keep members safe](https://www.linkedin.com/blog/engineering/trust-and-safety/how-our-content-abuse-defense-systems-work-to-keep-members-safe)
- [Fraud Detection Using Random Forest, Neural Autoencoder, and Isolation Forest Techniques](https://www.infoq.com/articles/fraud-detection-random-forest/?topicPageSponsorship=ed11260b-6513-40ba-922f-aae7ac9f942c)
- [智能风控筑基手册：全面了解风控策略体系 - 正阳的文章 - 知乎](https://zhuanlan.zhihu.com/p/151299288)
- [AITM](https://tech.meituan.com/2021/08/12/kdd-2021-aitm.html)
- [From shallow to deep learning in fraud](https://eng.lyft.com/from-shallow-to-deep-learning-in-fraud-9dafcbcef743)
