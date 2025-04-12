# Top-K

QPS会很大，但是总数N不会很多，内存可以轻松存下来

考点就是real time rough estimate vs offline accurate result

计算精确值的时候可以用HDFS或者Cassandra存每小时或者每天的数据，一个定时的map reduce job来根据需求计算这段时间内的top k(可以细化到秒)。长期的metadata可以直接删除也可以migrate到成本更低的S3里面。统计的top k result根据需求可以丢不同的数据库里。这里常问的点就是ingest这么多数据的时候如何来避免写过热。
Real time具体看QPS的大小，标准解法就是每个service直接把需要统计的项丢一个Kafka topic里面用Kafka来decouple ingest, 然后real time Kafka + Flink来统计每秒top k。但是要是QPS太大比如1M左右，其实real time也可以用batch来做in memory batch update的，每个server用in memory的hashmap来存key和counts，每200ms 提交一次in memory的这个result到 Redis里面然后另外每秒对Redis做real time Top K计算。我一般都会把这两种解法抛出来然后讨论下trade off。如果有需求做real time + historical top K 的话 real time计算出来的结果不能删除得和之前的甚至long term的结果结合起来做一个联合的top K。

## Reference

- https://www.1point3acres.com/bbs/thread-953468-1-1.html
- https://github.com/apssouza22/big-data-pipeline-lambda-arch
- https://serhatgiydiren.com/system-design-interview-top-k-problem-heavy-hitters/
-
