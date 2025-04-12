# 自然语言处理

NLP包括自然语言理解和自然语言生成，任务包括文本分类、相似匹配、对话问答、机器翻译、序列标注、知识图谱、意图理解、词义消歧。

- 视觉和nlp最大的区别：语义稀疏性，域间差异性，无限粒度性
- Transformer时代三类模型：bert（自编码）、gpt（自回归）、bart（编码-解码）

## 1. Tokenizer

- tokenizer: 大致经历了从word/char到subword的进化
- word level
  - 词表的长尾效应非常大，OOV问题，单词的形态关系和词缀关系(old, older)
- char level
  - 无法承载丰富的语义，序列长度长
- sub-word level: BPE, Bytes BPE, WordPiece, Uni-gram, SentencePiece
  - 常用词保持原状，生僻词应该拆分成子词以共享token压缩空间
  - BPE: byte-pair encoding 无监督分词，自底向上的策略。初始化将每个字符为词典，统计词频，迭代(合并频率最高的词对，更新词频)
  - wordpiece: 无监督分词，自顶向下的贪心拆分策略，最大似然估计确定最佳分割点(基于概率生成新subword)，词频更新词典
  - SentencePiece库: 基于BPE和uni-gram,根据不同任务或语料库需求，自定义分词模型，更好处理未登录或稀有词
  - chatGPT训练中文: BPE算法在中文上训，最小单元不再是汉字，而是 byte，UTF-8 编码中，一个汉字相当 3 个字节
  - 解决OOV(out-of-vocabulary)问题，even if a word is not seen during training, the model can still understand and generate text based on its constituent parts

## 2. 模型

**传统**

- BOW
- tfidf
- word2vec
- crf
- n-gram
  - [https://web.stanford.edu/~jurafsky/slp3/3.pdf](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

**encoder-decoder**

- BART
- T5

**encoder**

- BERT
- XLNet

**decoder**

- GPT3
- PALM
- LLaMA

### word2vec/glove/fasttext

- word2vec: 本质上是词的共现
- 缺点:
  - 静态表征(contextless word embeddings). 训练完成做推理时, 每个token的表示与上下文无关
  - 一词多义：disambiguate words with multiple meanings
- Hierarchical Softmax: 霍夫曼树， n分类变成 log(n) 次二分类
- Negative Sampling 负采样
  - 基于词频的采样，w^(0.75)
  - 负样本中选取一部分来更新，而不是更新全部的权重

### Transformer

- Transformer时代几大模型范式, BERT: encoder-only, GPT: decoder-only, T5: encoder-decoder, GLM: prefix-lm
- 预训练任务：Masked Language Model 和 Next Sentence Predict(Autoregressive)
- bert下游任务
- fine tune
  - adapter
- ELMo是分别以P(wi|w1...wi-1) 和 P(wi|wi+1...wn) 作为目标函数，独立训练处两个representation然后拼接，而BERT则是以p(wi|1..wi-1,wi+1,..wn) 作为目标函数训练LM。
- 位置编码
  - 绝对位置编码
  - 相对位置编码
  - 旋转位置编码RoPE
    - [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)
    - [ROUND AND ROUND WE GO! WHAT MAKES ROTARY POSITIONAL ENCODINGS USEFUL?](https://arxiv.org/pdf/2410.06205)

**ERNIE**

**RoBERTa**

- 调了更好的版本的BERT
- 预训练，无NSP任务
- 动态Mask: 训练数据复制多份，每份采用不同的随机挑选 token 进行掩码
- 更大的词汇表

**XLNet**

- 自回归

**UniLM**

**TinyBERT**

- Loss: Embedding Layer Distillation, Transformer Layer Distillation, Prediction Layer Distillation

### GPT

- 自回归模型
- GPT3: Zero-Shot Learning
- gpt的四个下游任务
- Emergent Ability

## 3. 评价指标

**perplexity**

**BLEU**

- BLEU 根据精确率(Precision)衡量翻译的质量，而 ROUGE 根据召回率(Recall)衡量翻译的质量
- 过于依赖参考，如果译文质量很好但部分字词在参考翻译中没有的话得分会很低；未考虑语法问题

**ROGUE (Recall-Oriented Understudy for Gisting Evaluation)**

- [基于召回率](https://zhuanlan.zhihu.com/p/647310970)
- ROUGE-N: N-gram拆分后，计算召回率
- ROUGE-L: 最长公共子序列(非连续)
- ROUGE-W: 连续匹配情况加权后的最长公共子序列长度

**BertScore**

## 4. 应用

> 对具体的应用方向应该建立和熟悉其发展脉络

### 4.1 文本分类

知识点

- word2vec
  - HS是怎么做的？负采样怎么做的？
  - 负采样：加速了模型计算,保证了模型训练的效果。模型每次只需要更新采样的词的权重，不用更新所有的权重，那样会很慢。中心词其实只跟它周围的词有关系，位置离着很远的词没有关系，也没必要同时训练更新。采样 ，词频越大，被采样的概率就越大。
- fasttext
  - fasttext的架构和word2vec 中的cbow 类似，不同之处在于，fasttext预测的是标签，cbow 预测的是中间词
  - 将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类
- bert
- albert
  - ALBERT如何节约参数和训练（SOP）不一样的点
- roberta
  - 动态mask，ratio

### 4.2 实体识别

**信息抽取（Information Extraction）**

- 序列标注（Sequence Labeling）
- 指针网络（Pointer Network）
  - PointerNet
  - UIE: 基于 prompt 的指针网络

**实体识别 NER**

- Nested NER/ Flat NER
- lower layers of a pre-trained LLM tend to reflect “syntax” while higher levels tend to reflect “semantics”
- CRF 是一个序列化标注算法，接收一个输入序列，输出目标序列，可以被看作是一个 Seq2Seq 模型

**关系抽取 RE**

- spert/ CasRel/[TPLinker](https://arxiv.org/abs/2010.13415)/GPLinker
- 关系抽取后的结果：保存Neo4j
- 嵌套->GP, 非连续->W2ner, 带prompt->UIE

**事件抽取 EE**

- djhee 和 plmee

**Entity Linking**

### 4.3 文本摘要 Text summarization

- 分为抽象式摘要（abstractive summarization）和抽取式摘要(extractive summarization)
- 在抽象式摘要中，目标摘要所包含的词或短语会不在原文中，通常需要进行文本重写等操作进行生成；
- 抽取式摘要，通过复制和重组文档中最重要的内容(一般为句子)来形成摘要。那么如何获取并选择文档中重要句子，就是抽取式摘要的关键。传统抽取式摘要方法包括Lead-3和TextRank，传统深度学习方法一般采用LSTM或GRU模型进行重要句子的判断与选择，可以采用预训练语言模型自编码BERT/自回归GPT进行抽取式摘要。

**常用指标**

- ROUGE
- BLEU

### 4.4 关键词提取

key phrase generation

- https://www.zhihu.com/question/21104071

- NPChunker

### 4.5 文本生成

- beam search

## 5. 解决问题

### 5.1 多语言模型 Multilingual

语言模型

- 语言模型的常用评价指标是困惑度perplexity
- 为多语言训练SentencePiece (SPM)

### 5.2 长序列

## 6. 问答

- 为啥文本不用batch norm要用layer norm
  - BN: batch之间每一个element之间的分布，对Batch Size大小很敏感; LN: 每一个example序列之间的分布标准化
  - [Rethinking Batch Normalization in Transformers](https://arxiv.org/abs/2003.07845)
- transformer计算kvq的含义
- 如何‌估计微调一个language model的成本是多少
- quantization的概念，解释一下如何工作的
- 如果文本非常长怎么处理
- 如何克服固定context window的限制，能不能有100K的context window
- Bert是怎么解决OOV问题
  - 如果一个单词不在词表中，则按照subword的方式逐个拆分token，如果连逐个token都找不到，则直接分配为[unknown]; WordPiece广泛覆盖，这种情况较少发生
- BERT/GPT的区别
  - decoder_only 模型通过逐步生成的方式处理信息，不会将信息**压缩**到单个表示中。
  - BERT 则通过 CLS token 将信息汇总到一个单一的表示中，这种压缩的方式用于处理下游任务。
  - 随着大模型时代，即使是传统NLP任务，在few shot或语义复杂场景的时候，GPT更有优势
- adam/adamW区别
- query理解
  - NER 品牌、品类等
  - 构建实体库
  - 提升：增强，构造邻居词，共现的实体补充文本
- NER 和 POS 任务有什么区别和相似
- 文本流利度的指标
- 生成
  - beam search: 累积概率最大的k个序列

## 参考

**精读**

- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)

**扩展**

- [n-gram in hadoop map-reduce](https://github.com/cloudera/python-ngrams/tree/master/native/src/main/java)
- [秒懂词向量Word2vec的本质 - 穆文的文章 - 知乎](https://zhuanlan.zhihu.com/p/26306795)
- [语言模型](https://zhuanlan.zhihu.com/p/90741508)
- [NLP 任务中有哪些巧妙的 idea？ - 邱锡鹏的回答 - 知乎](https://www.zhihu.com/question/356132676/answer/901244271)
- [Text clustering with K-means and tf-idf](https://medium.com/@MSalnikov/text-clustering-with-k-means-and-tf-idf-f099bcf95183)
- [https://github.com/firechecking/CleanTransformer](https://github.com/firechecking/CleanTransformer)
- [史上最细节的自然语言处理NLP/Transformer/BERT/Attention面试问题与答案 - 海晨威的文章 - 知乎](https://zhuanlan.zhihu.com/p/348373259)
- [https://github.com/deborausujono/word2vecpy](https://github.com/deborausujono/word2vecpy)
- [information-retrieval-book](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
- [The Evolution of Tokenization – Byte Pair Encoding in NLP](https://www.freecodecamp.org/news/evolution-of-tokenization/)
- [前处理 Tokenizer- Byte Pair Encoder](https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py)
- [【LLM拆了再装】 Tokenizer篇 - coreyzhong的文章 - 知乎](https://zhuanlan.zhihu.com/p/700283095)
- [Transformer学习笔记一：Positional Encoding（位置编码） - 猛猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/454482273)
- [https://github.com/wangle1218/KBQA-for-Diagnosis](https://github.com/wangle1218/KBQA-for-Diagnosis)
- [https://github.com/wangle1218/faq-qa-sys-v2](https://github.com/wangle1218/faq-qa-sys-v2)
- [beam search的简单实现（面试版） - lumino的文章 - 知乎](https://zhuanlan.zhihu.com/p/623540053)
- [LLM+Embedding构建问答系统的局限性及优化方案](https://zhuanlan.zhihu.com/p/641132245)
- [Generating N-grams from Sentences in Python](https://albertauyeung.github.io/2018/06/03/generating-ngrams.html/)
- [超长文本综述：Effective Long Context Scaling of Foundation Models](https://arxiv.org/pdf/2309.16039.pdf)
- [Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data](https://aclanthology.org/2020.acl-main.463/)
- [实体命名识别（NER）如何入门？ - 致Great的回答 - 知乎](https://www.zhihu.com/question/455063660/answer/2371455632)
- [流水的NLP铁打的NER：命名实体识别实践与探索 - 王岳王院长的文章 - 知乎](https://zhuanlan.zhihu.com/p/166496466)
- [Knowledge Graph & NLP Tutorial-(BERT,spaCy,NLTK)](https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk)
- [跨语言预训练模型简述 - 潘小小的文章 - 知乎](https://zhuanlan.zhihu.com/p/102045011)
- [Shopee Products Matching: Text Part ](https://www.kaggle.com/code/finlay/shopee-products-matching-text-part-english)
- [7篇文章弄清楚关系抽取的经典范式 - 眼睛里进砖头了的文章 - 知乎](https://zhuanlan.zhihu.com/p/480408779)
- [huggingface transformers教程总结 - 屯屯的文章 - 知乎](https://zhuanlan.zhihu.com/p/576691638)
- [Pytorch data Samplers & Sequence bucketing](https://www.kaggle.com/code/shahules/guide-pytorch-data-samplers-sequence-bucketing)
- [Pytorch BERT beginner's room](https://www.kaggle.com/code/chumajin/pytorch-bert-beginner-s-room)
- [https://transformers.run/c3/2022-03-18-transformers-note-6/](https://transformers.run/c3/2022-03-18-transformers-note-6/)
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
- [Bert/Transformer 被忽视的细节（或许可以用来做面试题） - LiteAI的文章 - 知乎](https://zhuanlan.zhihu.com/p/613407791)
- [在BERT已经成为NLP基础知识的当下，你会在面试中问关于BERT的什么问题? - Elesdspline的回答 - 知乎](https://www.zhihu.com/question/424003949/answer/2349589527)
- [NLP领域，你推荐哪些综述性的文章？](https://www.zhihu.com/question/355125622)

**课程**

- [邱锡鹏: nlp-beginner](https://github.com/FudanNLP/nlp-beginner)
- [CS224N](https://web.stanford.edu/class/cs224n/index.html#schedule)
- [Stanford CS25: V2 I Introduction to Transformers w/ Andrej Karpathy]()
