# Online learning


## 案例

行为数据logging -> 消息队列 -> 特征pipeline -> 消息队列 -> 训练服务器


## QA
- 推荐系统在线学习时如何应对延迟反馈(delayed feedback)问题?
  - 历史数据进行标签修正
  - 建立一个延迟模型，预测每个样本的反馈到达时间分布。基于预测的延迟时间，可以对实时数据进行推断和修正。
  - 使用生存分析方法预测某一行为（如点击或购买）是否会发生以及何时发生。对于未发生反馈的样本，可以用生存概率进行估计，避免简单地将其视为负样本。


## Reference
- [蘑菇街首页推荐视频流——增量学习与wide&deepFM实践（工程+算法） - 琦琦的文章 - 知乎](https://zhuanlan.zhihu.com/p/212647751)
- [支持在线模型更新的大型推荐系统](https://openmlsys.github.io/chapter_recommender_system/case_study.html)
- https://github.com/online-ml/river
