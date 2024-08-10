# 机器学习系统设计

ML design的核心，本质是训练一个**model**来实现某个任务，如prediction/ranking/classification
- 建模design, 包括优化目标，feature，data，模型结构，评价标准等
- 系统design, 偏重于在线serve大模型，包括feature store, ANN, ETL pipeline, MLOps等

**例子**
  - youtube recommendation/doordash search box/auto suggestion
  - design youtube violent content detection system
  - detecting unsafe content
  - design a monitoring system to realtime measure ML models, including features, score distribution, qps
  - abusive user detection

**业务目标**
  - improve engagement on a feed
  - improve customer churn
  - return items from search engine query

**名词解释**
  - 曝光(impression): 文档被用户看到
  - 点击率(click-through-rate，CTR): 文档d曝光的前提下，用户点击d的概率
  - 交互行为(engagement): 在点击的前提下, 文档的点赞、收藏、转发、关注作者、评论。电商的加购物车、下单、付款


## 面试过程

### 回答框架

- **明确需求**
  - 场景，功能，目标(engagement Or revenue)，约束，如何转化为机器学习问题(如推荐转化为二分类模型和原因)
- **数据**
  - positive label and negative label
  - scale of the system, user和item有哪些数据和量级，一些可做特征的数据是否有log
  - 从2个方面identify data：training + label, testing + ground truth
  - label来源: 从交互中收集, 人工标注, 人工标注加无监督辅助, 增强数据
  - 数据探讨: bias, 非均衡, label质量
  - GDPR/privacy，数据脱敏，数据加密
  - train/test data和product上distribution不一样怎么办
  - data distribution随时间改变怎么办
- **特征工程**
  - user, item and cross, context
  - sparse and dense feature
  - 实际工作中，每个ML组都有自己不同的embedding set。互相使用别人的embedding set。怎么pre-train, fine-train, 怎么combine feature非常重要
  - feature的ABtest怎么做？不同traffic做
- **模型**
  - 模型选择，考虑系统方面的constraint。比如prediction latency， memory。怎么合理的牺牲模型的性能以换取constraint方面的benefit
  - 每个design的选择，像平时写design doc一样比较不同选项的优劣
  - 大多数场景，模型之外都需要额外的策略兜底
- **评价**
  - offline and online
  - 模型的评价，比如：点击，转化，是否有广告？考察的是GMV，还是转化订单？
- **部署**
  - server or device
  - all users or a part of users
  - statically, dynamically(server or device) or model streaming
- **serving**
  - batch prediction or online prediction
- **monitoring**
  - 监控latency，QPS，precision，recall等参数
- **maintain**
  - retrain strategy


### 过程

- 一边白板画框图，一边告知面试官我要讲某几个部分
- 整个过程，讲清楚主题之前，不要陷入任何一部分的细节挖掘。随着问题介绍，data和细节都会明确
- 每个部分，尤其是自己熟悉的方面，要主动讲，因为每个部分都很重要
- move之前最后确认：Is there anywhere that you feel I missed?


## 问答

- how to scale
  - Scaling general SW system (distributed servers, load balancer, sharding, replication, caching, etc)
  - Train data / KB partitioning
  - Distributed ML
  - Data parallelism (for training)
  - Model parallelism (for training, inference)
  - Asynchronous SGD
  - Synchronous SGD
  - Distributed training
  - Data parallel DT, RPC based DT
  - Scaling data collection
  - machine translation for 1000 languages
    - NLLB
  - [embedding-> Deep Hash Embedding](https://zhuanlan.zhihu.com/p/397600084)
- Monitoring, failure tolerance, updating (below)
- Auto ML (soft: HP tuning, hard: arch search (NAS))
- 线上线下不一致
  - [推荐系统有哪些坑？](https://www.zhihu.com/question/28247353/answer/2126590086)
- 不同的数据用什么方式存储
- data pipeline怎么设计
- serving
  - Online A/B testing
    - Based on the online metrics we would select a significance level 𝛼 and power threshold 1 – 𝛽
    - Calculate the required sample size per variation
      - The required sample size depends on 𝛼, 𝛽, and the MDE Minimum Detectable Effect – the target relative minimum increase over the baseline that should be observed from a test
    - Randomly assign users into control and treatment groups (discuss with the interviewer whether we will split the candidates on the user level or the request level)
    - Measure and analyze results using the appropriate test. Also, we should ensure that the model does not have any biases.
  - If we are serving batch features they have to be handled offline and served at real time so we have to have daily/weekly jobs for generating this data.
  - If we are serving real time features then they need to be fetched/derived at request time and we need to be aware of scalability or latency issues (load balancing), we may need to create a feature store to lookup features at serve time and maybe some caching depending on the use case.
  - Where to run inference: if we run the model on the user’s phone/computer then it would use their memory/battery but latency would be quick, on the other hand, if we store the model on our own service we increase latency and privacy concerns but removes the burden of taking up memory and battery on the user’s device.
  - how often we would retrain the model. Some models need to be retrained every day, some every week and others monthly/yearly. Always discuss the pros and cons of the retraining regime you choose
- deploy
  - model serving是典型的low latency high qps
  - 负载均衡和自动伸缩：
  - latency如何优化
  - 这么多server如何deploy，以及如何push新的model version，在更新的时候如何保证qps不degrade

- Monitoring Performance
  - Latency (P99 latency every X minutes)
  - Biases and misuses of your model
  - Performance Drop
  - Data Drift
  - CPU load
  - Memory Usage


## 参考

- [https://github.com/ByteByteGoHq/ml-bytebytego](https://github.com/ByteByteGoHq/ml-bytebytego)
- [https://developers.google.com/machine-learning/recommendation](https://developers.google.com/machine-learning/recommendation)
- [https://research.facebook.com/blog/2018/5/the-facebook-field-guide-to-machine-learning-video-series/](https://research.facebook.com/blog/2018/5/the-facebook-field-guide-to-machine-learning-video-series/)
- [https://github.com/khangich/machine-learning-interview](https://github.com/khangich/machine-learning-interview)
- [Machine Learning Engineering by Andriy Burkov](https://www.amazon.com/Machine-Learning-Engineering-Andriy-Burkov/dp/1999579577)
- [https://github.com/shibuiwilliam/ml-system-in-actions](https://github.com/shibuiwilliam/ml-system-in-actions)
- [https://github.com/mercari/ml-system-design-pattern](https://github.com/mercari/ml-system-design-pattern)
- [https://github.com/chiphuyen/machine-learning-systems-design](https://github.com/chiphuyen/machine-learning-systems-design)
- [https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md)
- [https://github.com/ibragim-bad/machine-learning-design-primer](https://github.com/ibragim-bad/machine-learning-design-primer)
- [Grokking the Machine Learning Interview](https://www.educative.io/courses/grokking-the-machine-learning-interview)
- [https://about.instagram.com/blog/engineering/designing-a-constrained-exploration-system](https://about.instagram.com/blog/engineering/designing-a-constrained-exploration-system)
- [https://www.educative.io/courses/grokking-the-machine-learning-interview](https://www.educative.io/courses/grokking-the-machine-learning-interview)
- [https://www.youtube.com/c/BitTiger](https://www.youtube.com/c/BitTiger)
- [ML system 入坑指南 - Fazzie的文章 - 知乎](https://zhuanlan.zhihu.com/p/608318764)
- [模型生产环境中的反馈与数据回流 - 想飞的石头的文章 - 知乎](https://zhuanlan.zhihu.com/p/493080131)
- [https://www.1point3acres.com/bbs/thread-901192-1-1.html](https://www.1point3acres.com/bbs/thread-901192-1-1.html)
- [kuhung/machine-learning-systems-design](https://github.com/kuhung/machine-learning-systems-design)
- [ML design 面试的答题模板，step by step-1point3acres](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=585908&ctid=230680)
- [30+公司 MLE 面试准备经验分享-1point3acres](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=499352&ctid=230680)
- [AI system](https://github.com/microsoft/AI-System)
- [Guideline google cloud](https://cloud.google.com/architecture/ai-ml?hl=en)
- [从大公司的博客里学最新机器学习](https://www.1point3acres.com/bbs/thread-893173-1-1.html)
- [浅谈ML Design推荐系统面试心得, ask me anything](https://www.1point3acres.com/bbs/thread-490321-1-1.html)
- [CS294: AI for Systems and Systems for AI](https://ucbrise.github.io/cs294-ai-sys-sp19/)
- [CSE 599W: Systems for ML](https://dlsys.cs.washington.edu/)
- [https://github.com/microsoft/AI-System](https://github.com/microsoft/AI-System)
- [https://github.com/eugeneyan/ml-design-docs](https://github.com/eugeneyan/ml-design-docs)
- [https://www.machinelearninginterviews.com/ml-design-template/](https://www.machinelearninginterviews.com/ml-design-template/)
- [https://github.com/Doragd/Algorithm-Practice-in-Industry](https://github.com/Doragd/Algorithm-Practice-in-Industry)
- [买它 MLE E6 昂赛过经](https://www.1point3acres.com/bbs/thread-1018808-1-1.html)
- [https://www.evidentlyai.com/ml-system-design](https://www.evidentlyai.com/ml-system-design)
