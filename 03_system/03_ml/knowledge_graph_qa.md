# 知识图谱问答

> 2024年后，更多参考[graphRAG设计](./rag.md)

## 1. requirements

**Functional Requirements:**

- Natural language question input
- Accurate answers based on knowledge graph information
- Support for complex multi-hop reasoning
- Handle different question types (factoid, relationship, comparison)

**Non-Functional Requirements:**

- Low latency (<500ms response time)
- High availability (99.9%)
- Scalability to handle large knowledge graphs
- Support multiple languages
- Privacy and security compliance

## 2. ML task & pipeline

![](../../.github/assets/03ml-qa-kbqa.png)

![](../../.github/assets/03ml-qa-kbqa2.png)

a) Question Understanding:

- Question type classification
- Entity recognition and linking
- Relation extraction
- Query intent classification

b) Graph Processing:

- Graph embedding generation
- Subgraph retrieval
- Path ranking

c) Answer Generation:

- Answer extraction/generation
- Confidence scoring
- Evidence compilation

## 3. data collection

**Sources:**

- Knowledge Graphs:
  - Wikidata
  - DBpedia
  - Domain-specific KGs
  - Company internal KGs
- Training Data:
  - WebQuestions
  - ComplexWebQuestions
  - LC-QuAD 2.0
  - KQA Pro
  - MetaQA

**Data Processing:**

- Entity normalization
- Relation alignment
- Graph completion
- Question-answer pair generation

## 4. feature

**Question Features:**

- BERT/RoBERTa embeddings
- Dependency parsing features
- Named entity mentions
- Question type indicators

**Graph Features:**

- Node embeddings (TransE, RotatE)
- Structural features
- Path features
- Subgraph features

## 5. model

## 6. evaluation

**Metrics:**

- Accuracy
- F1 Score
- Hits@K
- MRR (Mean Reciprocal Rank)
- Path validity
- Answer completeness
- Reasoning correctness

**Testing Approaches:**

- Unit tests for each component
- Integration tests
- A/B testing
- Human evaluation
- Adversarial testing

## 7. deployment & serving

Infrastructure:

- Containerization with Docker
- Kubernetes for orchestration
- GPU support for inference
- Load balancing
- Auto-scaling

## 8. monitor & maintenance

Monitoring:

- Model performance metrics
- System health metrics
- Error rates and types
- Latency distribution
- Resource utilization

## reference

- [https://github.com/shawnh2/QA-CivilAviationKG](https://github.com/shawnh2/QA-CivilAviationKG)
- [美团知识图谱问答技术实践与探索](https://tech.meituan.com/2021/11/03/knowledge-based-question-answering-in-meituan.html)
- [检索式对话系统在美团客服场景的探索与实践](https://tech.meituan.com/2022/11/03/retrieval-based-dialogue-system.html)
- [阿里小蜜：知识结构化推动智能客服升级](https://mp.weixin.qq.com/s/x9CkAyLKgLj7E7K1F2Q6iA)
- [阿里实时语音与智能对话](https://mp.weixin.qq.com/s/scvTTqApSr8SbCKRlUoz-g)
- [智能机器人在滴滴出行场景的技术探索](https://mp.weixin.qq.com/s/MSy8OHzR3avObmOq9uSSFQ)
- [QQ浏览器智能问答技术探索实践](https://mp.weixin.qq.com/s/nN0aSXQN_IyjIJ1mRT5s3w)
- [达摩院基于元学习的对话系统](https://mp.weixin.qq.com/s/Ji_-hTe5vwpnyu-whj3PXg)
- [大话知识图谱-意图识别和槽位填充](https://zhuanlan.zhihu.com/p/165963264)
