# 系统设计

1. 理解系统设计需求，需要明确系统所用于何种商业目的，要求的功能&技术，来成功定义面试官内心的“标答” 【功能性需求】
2. 明确所设计系统需求的资源, 例如分析QPS 【非功能性需求】
  - latency sensitive的场景，要避免使用msg queue
3. 画出关系清晰的架构图，搞清楚service怎么划分
  - high level design
4. 设计数据结构与存储、核心子服务、接口等，接上database
  - 技术选型：SQL vs No-SQL，Sync VS Async，RPC VS MQ等技术选择
5. 明确所设计系统的可扩展性、容错性、延迟要求等
  - 扩展: 加缓存，数据库读写分离，数据库 sharding 等等
  - 瓶颈: 一般都在数据库
6. 解决缺陷并处理可能遇到的问题


## Template
- Functional requirement
- NonFunctional requirement
- Capacity planning
- Diagram + API 【接口和业务entity】
- DataSchema + Scale
- Monitoring/Metrics


## 常见面试
- Design a URL shortener (e.g. Bitly)
- Design a video watching website (e.g. YouTube)
- Design a chatting service (e.g. Telegram, Slack, Discord)
- Design a file sharing service (e.g. Google Drive, Dropbox)
- Design a ride sharing service (e.g. Uber, Lyft)
- Design a photo sharing service (e.g. Flickr, Pinterest)
- Design an e-commerce website (e.g. Amazon, eBay)
- Design a jobs portal (e.g. LinkedIn, Indeed)
- Design a web crawler (e.g. Google)
- Design AI自动写作系统设计
- Design Auction system
- Design search autocomplete system
- Design large scale rate limiter
- Design Netflix
- Design Amazon inventory system
- Design Venmo
- Design large scale notification platform
- Design AirBNB platform
- Design Ticket master
- Design high velocity bank account platform
- Design top10 scorersina large scale mobilegame
- Design large scale real time chat platform
- Design electric bike rental platform
- Design grocery store order processing system
- Design website building platform
- Design giftcard system
- Design large scale devices location tracker
- auto complete: trie database


瓶颈，scale, tradeoff

## High level design
client -> load balancer -> web service/API -> memory cache -> DB

web/mobile -> HAproxy/ELB -> API Gateway -> Nginx server/Kube deployment/REST -> Redis -> MySQL

client -> reverse proxy -> web service -> message queue -> application server -> cache -> DB


## 问题
- How to do large scale batch processing
- How to do model serving readiness
- How to do model rollout
- 两个service要互相发消息，怎么解决
- 高并发系统
  - 缓存
  - 降级
  - 限流
    - 计数器、漏桶和令牌桶
- 负载均衡算法
  - 轮询、加权轮询、随机算法、一致性Hash
- 消息队列
  - 解耦，异步处理，削峰/限流
- 一致性
- 热点数据处理


## 基础

push and pull

consistent hashing

event sourcing

paxos and raft

cache

redis

数据库规范化(Normalization)
- Normalisation Form(NF)，其中包括第一范式、第二范式、第三范式、第四范式以及第五范式(1NF、2NF、3NF、4NF、5NF)
- SQL provides ACID transaction

网络基础

Restful API

HTTP/HTTPS

RPC

### SQL and NoSQL

需要支持transaction和join的，需要用SQL
需要high TPS和灵活schema的，用NoSQL

SQL(ACID)
- consistency
- structured data (fixed schema)
- transactions
- joins

NOSQL
- high performance
- unstructured data (flexible schema)
- availability
- easy scalability


### scale
- load balancer
- Horizontal scaling
- caching
- database sharding(partition)
- horizontal scaling, sharded by userId, caching, etc



## Reference
- Web Application and Software Architecture 101
- [Uber tech blog](https://www.uber.com/en-SE/blog/)
- [pinterests tech blog](https://medium.com/pinterest-engineering)
- [System design interview guide for Software Engineers](https://www.techinterviewhandbook.org/system-design/)
- [https://github.com/madd86/awesome-system-design](https://github.com/madd86/awesome-system-design)
- [https://github.com/binhnguyennus/awesome-scalability](https://github.com/binhnguyennus/awesome-scalability)
- [https://github.com/codersguild/System-Design](https://github.com/codersguild/System-Design)
- [https://github.com/soulmachine/system-design](https://github.com/soulmachine/system-design)
- [http://icyfenix.cn/](http://icyfenix.cn/)
- [https://github.com/InterviewReady/system-design-resources](https://github.com/InterviewReady/system-design-resources)
- [https://rajat19.github.io/system-design/pages/guide.html](https://rajat19.github.io/system-design/pages/guide.html)
- [https://time.geekbang.org/column/article/6458](https://time.geekbang.org/column/article/6458)
- [https://blog.bytebytego.com/?sort=top](https://blog.bytebytego.com/?sort=top)
- [https://soulmachine.gitbooks.io/system-design/content/cn/](https://soulmachine.gitbooks.io/system-design/content/cn/)
- [System Design Introduction For Interview.](https://www.youtube.com/watch?v=UzLMhqg3_Wc)
- [crack-the-system-design-interview](https://tianpan.co/notes/2016-02-13-crack-the-system-design-interview)
- [https://github.com/imkgarg/Awesome-Software-Engineering-Interview/blob/master/SystemDesign.md](https://github.com/imkgarg/Awesome-Software-Engineering-Interview/blob/master/SystemDesign.md)
- [https://tianpan.co/notes/2016-02-13-crack-the-system-design-interview](https://tianpan.co/notes/2016-02-13-crack-the-system-design-interview)
- [滴滴专车业务与技术流程](http://alexstocks.github.io/html/didi.html)
- [从无到有：微信后台系统的演进之路](https://www.infoq.cn/article/the-road-of-the-growth-weixin-background/)
- [如何设计微信的聊天系统? - 土汪的文章 - 知乎](https://zhuanlan.zhihu.com/p/34369396)
- [https://blog.twitter.com/engineering/en_us/a/2014/building-a-complete-tweet-index](https://blog.twitter.com/engineering/en_us/a/2014/building-a-complete-tweet-index)
- [在线广告系统----计算机王冠上的明珠](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=890389&ctid=9)
- [RESTful Service API 设计](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=909427&ctid=9)
- [我是如何准备系统设计的](https://www.1point3acres.com/bbs/thread-660847-1-1.html)
- [题目类型](https://codemia.io/system-design)
- [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832)
- [Design fb-live-comments](https://www.hellointerview.com/learn/system-design/answer-keys/fb-live-comments)
- [Dynamo: Amazon’s Highly Available Key-value Store](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
- 