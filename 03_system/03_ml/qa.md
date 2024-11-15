# 问答系统

- Information Retrieval: 基于问答库，利用信息检索(召回、排序)从问答库中选择查询问题与已有问题和答案中最相似的作为输出
- Knowledge Based Question Answering: 基于知识库，围绕知识库的实体或者关系，把问题转化成数据库上的查询语句，然接查库获取答案
- Task-Bot: 定义场景下一共需要哪些槽值（Slot），用户对话中提到了槽值，抽取记录下来，如果没提到的，就反问，直到槽值都填满了进行查询给出操作
- generative/Seq2seq Bot: 完全生成式模型


## 1. requirements
- 闲聊还是任务(业务型，知识型)
- Close-domain QA 还是 Open-domain QA (海量文档，来回答一个事实性问题factoid questions)
- 单轮(single-turn)还是多轮(multi-turn)
- 闭卷问答(closed-book QA)还是基于上下文的问答(in-context learning QA)
- 多领域
- 多语言
- non-functional requirement: **latency** 和 **throughput**


## 2. ML task & pipeline
- 检索式（Retrieval），生成式（Generative），任务式
  - 检索式：主要思路是从对话语料库中找出与输入语句最匹配的回复，这些回复通常是预先存储的数据。
  - 生成式：主要思路是基于深度学习的Encoder-Decoder架构，从大量语料中习得语言能力，根据问题内容及相关实时状态信息直接生成回答话术。
  - 任务式：就是任务型对话，通常要维护一个对话状态，根据不同的对话状态决策下一步动作，是查询数据库还是回复用户等等。
- 知识图谱：爬取的数据经过关系抽取存入Neo4j数据库
- 基于本地知识的问答库，Bert微调，建立倒排 (inverted index)，特征向量余弦相似度
- GPT

### 2.1 任务型
管道方法
- 自然语言理解(领域识别 domain，意图识别 intents，语意槽填充 slots)
- 对话管理 Dialogue Management(状态追踪，对话策略优化，知识库与API)
- 自然语言生成


端对端


## 3. data collection


## 4. model

### 4.1 问题理解
- 领域/意图识别
- 实体识别
- 槽位填充


### 4.2 召回
- 倒排索引（Inverted Index）和近似近邻搜索（ApproximateNearest Neighbor Search）进行快速检索



## 5. evaluation
- 传统生成指标n-gram based metrics (BLEU、ROUGE)
- 基于语义距离的指标 BERT-Score
- MT-bench和Chatbot Arena进行人工排序
- 使用GPT4等模型进行打分


## 6. deploy & serving


## 问答
- 如何处理业务对话系统中的unexpected intent
  - 可继续阅读参考中rasa中的：Unexpected Intent Policy


## 参考
- [对话系统与四大天王 - 王岳王院长的文章 - 知乎](https://zhuanlan.zhihu.com/p/358001553)
- [QA survey](https://github.com/BDBC-KG-NLP/QA-Survey-CN)
- [美团智能客服核心技术与实践](https://tech.meituan.com/2021/09/30/artificial-intelligence-customer-service.html)
- [Intelligent Automation Platform: Empowering Conversational AI and Beyond at Airbnb](https://medium.com/airbnb-engineering/intelligent-automation-platform-empowering-conversational-ai-and-beyond-at-airbnb-869c44833ff2)
- [Task-Oriented Conversational AI in Airbnb Customer Support](https://medium.com/airbnb-engineering/task-oriented-conversational-ai-in-airbnb-customer-support-5ebf49169eaa)
- [面向领域应用的大模型关键技术](https://mp.weixin.qq.com/s/l91izY8GBFsyyPgiSPHU6w)
- [Open source machine learning framework: https://github.com/RasaHQ/rasa](https://github.com/RasaHQ/rasa)
- [RAG探索之路的血泪史及曙光 - 小虫飞飞的文章 - 知乎](https://zhuanlan.zhihu.com/p/664921095)
- [Building a Question Answering System Part 1: Query Understanding in 18 lines of Code](https://medium.com/casl-project/building-a-question-answering-system-part-1-query-understanding-in-18-lines-916110f9f2b2)
- [基于知识增强和预训练大模型的 Query 意图识别](https://mp.weixin.qq.com/s/lVGKwNDgaHLROPdN3XUmiw)
- [NLP多轮对话如果做得足够好，会有哪些明朗的落地应用？ - 袋鼠猪的回答 - 知乎](https://www.zhihu.com/question/474271324/answer/2629631795)
- [CMU11492-语音](https://espnet.github.io/espnet/notebook/)
- [2018-从零开始搭建智能客服](https://www.sohu.com/a/228122295_355140)