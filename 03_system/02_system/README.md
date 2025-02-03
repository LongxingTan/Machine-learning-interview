# 系统设计

- 理解系统设计需求，需要明确系统所用于何种商业目的，要求的功能&技术，来成功定义面试官内心的“标答” 【功能性需求】
- 明确所设计系统需求的资源, 例如分析QPS 【非功能性需求】
  - latency sensitive的场景，要避免使用msg queue
- 画出关系清晰的架构图，搞清楚service怎么划分
  - high level design
  - `现在我们面临什么问题，解决这个问题有哪几种思路，其优略分别是什么，鉴于此我更倾向于那种设计`
- 设计数据结构与存储、核心子服务、接口等，接上database
  - 技术选型：SQL vs No-SQL，Sync VS Async，RPC VS MQ等技术选择
- 明确所设计系统的可扩展性、容错性、延迟要求等
  - 扩展: 加缓存，数据库读写分离，数据库 sharding
  - 瓶颈: 一般在数据库
- 解决缺陷并处理可能遇到的问题
  - consistency: Lock + Time To Live
  - large traffic: Message queue


## Template
- Functional requirement
- NonFunctional requirement
  - summarize系统重点: read or write heavy, consistency or availability more important
- Capacity planning
- Diagram + API 【接口和业务entity】
- DataSchema + Scale
- Monitoring/Metrics


## 常见面试
> Write Heavy System or Read Heavy System, Strong Consistency System, consistency/ availability priority, Scheduler System
- [Design a URL shortener (e.g. Bitly)](./tinyurl.md)
- [AD Click Aggregator](./ad_click_aggregator.md)
- [Web Crawler (e.g. Google)](./crawler.md)
- [Design large scale rate limiter](./rate_limiter.md)
- Design Auction system
- Ticketmaster
- LeetCode
- Leaderboard
- Design a video watching website (e.g. YouTube)
- Design a chatting service (e.g. Telegram, Slack, Discord)
- Design a file sharing service (e.g. Google Drive, Dropbox)
- Design a ride sharing service (e.g. Uber, Lyft)
- Design a photo sharing service (e.g. Flickr, Pinterest)
- Design an e-commerce website (e.g. Amazon, eBay)
- Design a jobs portal (e.g. LinkedIn, Indeed)
- Design AI自动写作系统设计
- Design search autocomplete system
- Design Netflix
- Design Amazon inventory system
- Design Venmo
- Design large scale notification platform
- Design AirBNB platform
- Design Ticket master
- Design high velocity bank account platform
- Design top10 scorer sina large scale mobile game
- Design large scale real time chat platform
- Design electric bike rental platform
- Design grocery store order processing system
- Design website building platform
- Design gift card system
- Design large scale devices location tracker
- Top K


## High level design
> 瓶颈，scale, tradeoff

client -> load balancer -> web service/API -> memory cache -> DB

web/mobile -> HAproxy/ELB -> API Gateway -> Nginx server/Kube deployment/REST -> Redis -> MySQL

client -> reverse proxy -> web service -> message queue -> application server -> cache -> DB


## 基础

### 数据库基础
**数据库规范化(Normalization)**
- Normalisation Form(NF)，其中包括第一范式、第二范式、第三范式、第四范式以及第五范式(1NF、2NF、3NF、4NF、5NF)
- SQL provides ACID transaction

**SQL**
需要支持transaction和join的
- SQL(ACID)
- consistency
- structured data (fixed schema)
- transactions
- joins

**NoSQL**
需要high TPS和灵活schema的
- high performance
- unstructured data (flexible schema)
- availability
- easy scalability, 分布式架构在 NoSQL 数据库中非常普遍


### 网络基础

Restful API

HTTP/HTTPS
- HTTP是基于TCP/IP协议的应用层协议，定义的是传输数据的内容规范

RPC
- 解决分布式系统中，服务之间的调用问题；远程调用时，让调用者感知不到远程调用的逻辑
- RPC架构的核心组件: Client, Server, Client Stub, Server Stub, stub理解为存根

socket
- Socket不属于协议，而是一个调用接口（API），属于网络协议的传输层，是对TCP/IP协议的封装
- socket长链接: 长连接多用于操作频繁，点对点的通讯，而且连接数不能太多情况；如推送，聊天，保持心跳长连接等

TCP
- 建立连接需要经过三次握手，断开连接需要经过四次分手


### 分布式基础
- load balancer
- Horizontal scaling
- caching
  - redis
- database sharding(partition)
- horizontal scaling, sharded by userId, caching, etc
- consistent hashing
- paxos and raft

### 消息队列基础
- [你就能明白kafka的工作原理了](https://zhuanlan.zhihu.com/p/68052232)
- https://medium.com/@andrew_schofield/queues-for-kafka-29afa8aeed86


## 问题
- How to do scale
  - 数据分区（Sharding）
  - 功能分区（Functional Partitioning）
  - 添加副本（Replication）
- 高并发系统
  - 缓存
  - 降级
  - 限流: 计数器、漏桶和令牌桶
- How to do large scale batch processing
- How to do model serving readiness
- How to do model rollout
- 两个service要互相发消息，怎么解决
- 负载均衡算法
  - 轮询、加权轮询、随机算法、一致性Hash
- 消息队列
  - 解耦，异步处理，削峰/限流
- 一致性
- 热点数据处理
- 幂等 Idempotent


## Reference
**精读**
- [grokking-the-system-design-interview](https://www.educative.io/courses/grokking-the-system-design-interview)
- [system design primer](https://github.com/donnemartin/system-design-primer)
- [DDIA-Designing Data-Intensive Application](https://github.com/Vonng/ddia)
- [youtube-System Design Interview](https://www.youtube.com/@SystemDesignInterview)
- [youtube-System Design Guru](https://www.youtube.com/@newgpu-sys-design)
- [一篇文章解决所有system design面试](https://www.1point3acres.com/bbs/thread-559285-1-1.html)

**扩展**
- [How Slack Works](https://www.youtube.com/watch?v=WE9c9AZe-DY&list=PLndbWGuLoHeYTBaqFu31Nac-19qsdUl_V)
- Web Application and Software Architecture 101
- [Jordan has no life](https://www.youtube.com/@jordanhasnolife5163/videos)
- [https://blog.bytebytego.com/?sort=top](https://blog.bytebytego.com/?sort=top)
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
- [https://soulmachine.gitbooks.io/system-design/content/cn/](https://soulmachine.gitbooks.io/system-design/content/cn/)
- [System Design Introduction For Interview.](https://www.youtube.com/watch?v=UzLMhqg3_Wc)
- [crack-the-system-design-interview](https://tianpan.co/notes/2016-02-13-crack-the-system-design-interview)
- [https://github.com/imkgarg/Awesome-Software-Engineering-Interview/blob/master/SystemDesign.md](https://github.com/imkgarg/Awesome-Software-Engineering-Interview/blob/master/SystemDesign.md)
- [https://tianpan.co/notes/2016-02-13-crack-the-system-design-interview](https://tianpan.co/notes/2016-02-13-crack-the-system-design-interview)
- [滴滴专车业务与技术流程](http://alexstocks.github.io/html/didi.html)
- [https://blog.twitter.com/engineering/en_us/a/2014/building-a-complete-tweet-index](https://blog.twitter.com/engineering/en_us/a/2014/building-a-complete-tweet-index)
- [RESTful Service API 设计](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=909427&ctid=9)
- [我是如何准备系统设计的](https://www.1point3acres.com/bbs/thread-660847-1-1.html)
- [题目类型](https://codemia.io/system-design)
- [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832)
- [Design fb-live-comments](https://www.hellointerview.com/learn/system-design/answer-keys/fb-live-comments)
- [Dynamo: Amazon’s Highly Available Key-value Store](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
- [https://github.com/preslavmihaylov/booknotes/tree/master/system-design/system-design-interview](https://github.com/preslavmihaylov/booknotes/tree/master/system-design/system-design-interview)
- [https://engineeringblog.yelp.com/](https://engineeringblog.yelp.com/)
- https://github.com/luxu1220/redis_practice
- [L6和L5系统设计面试区别](https://www.1point3acres.com/bbs/thread-1054990-1-1.html)
- [System Design的个人见解和一些例子](https://www.1point3acres.com/bbs/thread-953468-1-1.html)
