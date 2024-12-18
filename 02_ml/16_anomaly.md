# 异常检测

- 异常类型：Point anomaly, Contextual anomaly, Collective anomaly
- 具体类型：根据取值范围、常数波动、固定斜率波动、滑动聚合加方差、一阶差分、局部异常
- 任务类型：有监督，无监督
- 模型类型：距离，分类


## 模型
**规则、经验**
- remove outlier: Random sample consensus (RANSAC) is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates


**3sigma**
- Z-score标准化，数据满足正态分布

**PCA**
- 降维

**孤立森林**

**One-Class SVM**

**DBSCAN**

**autoencoder**

**GAN**


## reference

- [Warden: Real Time Anomaly Detection at Pinterest](https://medium.com/pinterest-engineering/warden-real-time-anomaly-detection-at-pinterest-210c122f6afa)
- [Tubi 时间序列 KPI 的异常值检测 - Tubi 中国团队的文章 - 知乎](https://zhuanlan.zhihu.com/p/642174241)
- [基于AI算法的数据库异常监测系统的设计与实现](https://tech.meituan.com/2022/09/01/database-monitoring-based-on-ai.html)
- [网易如何做到数据指标异常发现和诊断分析？](https://mp.weixin.qq.com/s/wr9XvBNRBeKfp6acxkXc2A)
- [How our content abuse defense systems work to keep members safe](https://www.linkedin.com/blog/engineering/trust-and-safety/how-our-content-abuse-defense-systems-work-to-keep-members-safe)
- [Fraud Detection Using Random Forest, Neural Autoencoder, and Isolation Forest Techniques](https://www.infoq.com/articles/fraud-detection-random-forest/?topicPageSponsorship=ed11260b-6513-40ba-922f-aae7ac9f942c)