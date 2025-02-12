# RAG

大模型相关系统设计可参考[Microsoft graphRAG](https://github.com/microsoft/graphrag)，[Google notebookLM](https://notebooklm.google/?location=unsupported)，deep research


## 1. requirements
> freshness considered as correct answer may change over time

**use case**
- agent功能
- 多轮 multi turn follow-up
- 是否提供reference
- personalization

**constraint**
- latency
- Throughput
- Availability
- Scalability 


## 2. ML task & pipeline
- indexing
- retrieval
- generation

![](../../.github/assets/03ml-rag.jpeg)

**optional**
- rewrite
- routing


## 3. data
- document chunk
- data augmentation (e.g. query expansion)
- query rewrite 


## 4. model
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
- KV cache


## 7. monitoring & maintenance


## 8. 优化与问答

- NL2SQL
- 幻觉
- 如何单独更新知识库中某个文档？
  - 增量更新, 给文档添加版本号
- 多轮对话的RAG如何实现
  - 历史记录重写查询: 基于多轮的会话记录与当前问题，调用大模型生成一个新问题. llamaindex提供了CondenseQuestionChatEngine, ContextChatEngine


## Reference
- [https://github.com/langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)
- [Better RAG 1: Advanced Basics](https://huggingface.co/blog/hrishioa/retrieval-augmented-generation-1-basics)
- [https://github.com/langgenius/dify](https://github.com/langgenius/dify)
- [Mock_ML_System_Design_RAG_Chat_With_Doc_Versions](https://github.com/ML-SystemDesign/MLSystemDesign/tree/main/Design_Doc_Examples/Mock_ML_System_Design_RAG_Chat_With_Doc_Versions)
- [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)
- [Building a RAG Batch Inference Pipeline with Anyscale and Union](https://www.anyscale.com/blog/anyscale-union-batch-inference-pipeline)
- [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning)
- [How we built Text-to-SQL at Pinterest](https://medium.com/pinterest-engineering/how-we-built-text-to-sql-at-pinterest-30bad30dabff)