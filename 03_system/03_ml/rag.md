# RAG

大模型相关系统设计可参考[Microsoft graphRAG](https://github.com/microsoft/graphrag)，[Google notebookLM](https://notebooklm.google/?location=unsupported)，deep research

## 1. requirements

> freshness considered as correct answer may change over time

**use case**

- scope, product customer
- agent功能
- 多轮 multi turn follow-up
- 多跳 multi hop
- 是否提供reference
- personalization

**constraint**

- latency
- Throughput
- Availability
- Scalability
- security
  - Guardrails

**知识库**

- Is the data structured (e.g., databases, knowledge graphs) or unstructured (e.g., text documents, web pages)?
- Volume: How large is the knowledge base?
- Velocity: How frequently does the knowledge base change?
- Veracity: How reliable and trustworthy is the information?

**大模型推理**

- TTFT (Time to first token)
- TBT (Time between tokens)
- Throughput

## 2. ML task & pipeline

- indexing
- retrieval
  - 与[search技术](./search_engine.md)类似
- generation

![](../../.github/assets/03ml-rag.jpeg)

**optional**

- rewrite
- routing
- 人工审批流程
- memory
- fusion
- multimodal
- Long context

**scale challenge**
- 相比传统搜索，更多的向量检索，对GPU、data infra、storage更大的挑战


## 3. data

- document: structure and unstructured data
- document chunk
- data augmentation (e.g. query expansion)
- query rewrite

## 4. model
> DeepSearch 可以被视作一个配备了各类网络工具（比如搜索引擎和网页阅读器）的 LLM Agent。这个 Agent 通过分析当前的观察结果以及过往的操作记录，来决定下一步的行动方向：是直接给出答案，还是继续在网络上探索。这就构建了一种状态机架构，其中 LLM 负责控制状态间的转换。在每一个决策点，你都有两种可选方案：你可以精心设计提示词，让标准的生成模型产生特定的操作指令；或者，也可以利用像 Deepseek-r1 这样专门的推理模型，来自然而然地推导出下一步应该采取的行动。然而，即使使用了 r1，你也需要定期中断它的生成过程，将工具的输出结果（比如搜索结果、网页内容）注入到上下文之中，并提示它继续完成推理过程。

**retrieval**

- hybrid
- finetuning loss
- finetuning dataset prepare

**llm**

- finetuning
- prompt engineering

## 5. evaluation

**retrieval**

- mrr, ndcg
- relevance, coherence(连贯性)

**generation**

- rouge-l, 关键词重合度
- 主观评估：质量，准确性

**online**

- AB testing
- useful/truthful

## 6. deploy & service

- Tracing
  - 记录和监控各个组件的调用、性能、输入输出等。识别瓶颈、调试问题，并优化性能
- KV cache

## 7. monitoring & maintenance

## 8. 优化与问答

- NL2SQL
- 幻觉
  - [Chain-of-Verification](https://arxiv.org/abs/2309.11495)
- 如何单独更新知识库中某个文档？
  - 增量更新, 给文档添加版本号
- 多轮对话的RAG如何实现
  - 历史记录重写查询: 基于多轮的会话记录与当前问题，调用大模型生成一个新问题. llamaindex提供了CondenseQuestionChatEngine, ContextChatEngine
  - memory 模块
- 召回结果中有相互排斥的信息
- 如何输出参考引用来源
  - 为LLM提供来源 ID（如文档编号、段落ID、chunk哈希），模型在生成时可引用;
  - 将生成内容与原始文本 chunk 做后处理匹配
- 是否需要调用 RAG判断
  - 置信度驱动触发机制，模型输出每条回答的置信度，低于阈值则触发RAG检索
  - Self-Ask 模式，“我是否需要外部知识回答这个问题？”
  - 训练二分类器判断是否需检索外部知识

## Reference

- [https://github.com/langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)
- [Better RAG 1: Advanced Basics](https://huggingface.co/blog/hrishioa/retrieval-augmented-generation-1-basics)
- [https://github.com/langgenius/dify](https://github.com/langgenius/dify)
- [Mock_ML_System_Design_RAG_Chat_With_Doc_Versions](https://github.com/ML-SystemDesign/MLSystemDesign/tree/main/Design_Doc_Examples/Mock_ML_System_Design_RAG_Chat_With_Doc_Versions)
- [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)
- [Building a RAG Batch Inference Pipeline with Anyscale and Union](https://www.anyscale.com/blog/anyscale-union-batch-inference-pipeline)
- [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning)
- [How we built Text-to-SQL at Pinterest](https://medium.com/pinterest-engineering/how-we-built-text-to-sql-at-pinterest-30bad30dabff)
- [DeepSearch 与 DeepResearch 的设计和实现 - Jina AI的文章 - 知乎](https://zhuanlan.zhihu.com/p/26560000573)
