# 大数据

## Design
设计一个操作系统内存管理分页分段的API


给一个分布式计算拓扑图，类似hadoop的map reduce，讨论怎么能improve efficiency，可以讨论的点包括task scheduler，failover，hash consistency，at least/at most/exactly once, 网络, etc. 


## Basic

Flume: 采集用户行为数据

Sqoop: 迁移业务数据库

Spark SQL: 计算统计类标签

Azkaban: 定时执行计算任务


## Simple RDD 

```python
class InMemoryRDD:
    def __init__(self, data):
        self.data = data

    def map(self, func):
        """Applies a function to each element in the RDD."""
        return InMemoryRDD([func(x) for x in self.data])

    def filter(self, func):
        """Filters elements in the RDD based on a predicate."""
        return InMemoryRDD([x for x in self.data if func(x)])

    def flatMap(self, func):
        """Applies a function that returns an iterable and flattens the result."""
        return InMemoryRDD([item for x in self.data for item in func(x)])

    def collect(self):
        """Returns the data as a list."""
        return self.data

    def reduce(self, func):
        """Aggregates the elements of the RDD using a function."""
        from functools import reduce
        return reduce(func, self.data)

    def count(self):
        """Counts the number of elements in the RDD."""
        return len(self.data)

    def distinct(self):
        """Removes duplicate elements."""
        return InMemoryRDD(list(set(self.data)))
```


## 参考

- [udemy-mastering databricks & apache spark-build ETL data pipeline](https://www.bilibili.com/video/BV1LU4y1s7ac/)
- [请用通俗形象的语言解释下：Spark中的RDD到底是什么意思？ - 木鸟杂记的回答 - 知乎](https://www.zhihu.com/question/37437257/answer/2571373097)
- [https://github.com/brunoluz/the-ultimate-hands-on-hadoop-tame-your-big-data](https://github.com/brunoluz/the-ultimate-hands-on-hadoop-tame-your-big-data)
- [https://github.com/TurboWay/bigdata_analyse](https://github.com/TurboWay/bigdata_analyse)
- [https://github.com/heibaiying/BigData-Notes](https://github.com/heibaiying/BigData-Notes)
- [Google Colab 的正确使用姿势 - 佘城璐的文章 - 知乎](https://zhuanlan.zhihu.com/p/218133131)
- [databricks 使用spark](https://blog.csdn.net/RONE321/article/details/90413306)
- [https://www.1point3acres.com/bbs/thread-1061061-1-1.html](https://www.1point3acres.com/bbs/thread-1061061-1-1.html)
- [MapReduce: Simplified Data Processing on Large Clusters](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf)
- [Bigtable: A Distributed Storage System for Structured Data](https://static.googleusercontent.com/media/research.google.com/en//archive/bigtable-osdi06.pdf)
- [Big Data Analytics Options on AWS](https://docs.aws.amazon.com/whitepapers/latest/big-data-analytics-options/welcome.html)
- [Introduction to SQL in BigQuery](https://cloud.google.com/bigquery/docs/introduction-sql)