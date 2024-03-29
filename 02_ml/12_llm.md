# 大语言模型

## scaling law
- C(计算量) = 6 N(模型参数量) * D(数据集大小)


## 数据
- 多样性：不同情景、背景、语境
- 数量
- 质量：重要
- 采集方式：随机无偏差


## 模型

encoder-decoder：BERT，BART，T5，UL2
decoder-only：GPT3，Codex，PALM，Galactica，Chinchilla，LLaMA，OPT，BLOOM，Gopher


GPT，BERT，Transformer等

InstructGPT，LLama，ChatGLM

**LLaMa**
- llama的self-attention和mlp中没有bias
- 使用 rmsnorm而不是layernorm，少计算了均值


## 评测
- rouge
- BLEU
- perplexity


### SFT
in context learning

#### 高效参数微调 PEFT
- Prompt tuning
  - 固定模型前馈层参数，仅仅更新部分embedding参数即可

- Adapter Tuning
  - 所有参数微调较为低效，只微调下游的几层效果不佳。因此设计了adapter结构，在原结构中稍微增加了参数，只微调该部分，效果接近full-finetune
  - down-project层将高维度特征映射到低维特征，过一个非线形层之后，再 up-project 结构将低维特征映射回原来的高维特征. skip-connection 结构，确保在最差的情况下退化为 identity

- Prefix Tuning 前缀微调
  - 问题：最终的性能对人工设计的template的特别敏感，加一个词或者少一个词，或者变动位置，都会造成很大的变化
  - 使用连续的virtual token embedding来代替离散的token. 将一个连续的特定于任务的向量序列添加到输入，称之为前缀. Prefix是可以学习的“隐式”的提示，训练的时候只更新Prefix部分的参数，而Transformer中的预训练参数固定

- p-tuning
  - 自然语言的离散模版转化为可训练的隐式prompt

- LoRA 低秩自适应
  - 用低秩的方式（一个矩阵可以用两个较小的矩阵相乘来近似）来调整参数矩阵.
  - 冻结预训练好的模型权重参数, 通过往模型中加入额外的网络层，并只训练这些新增的网络层参数
  - QLoRA: Quantized LoRA, 使用QLoRA算法要结合bitsandbytes库和peft库

```python

```

### RLHF
RLHF 与 RLAIF

better align with human preferences and reduce undesired outcomes in scenarios

SFT负责Instruction following，RL强化helpfulness、honesty、safety偏好


Proximal Policy Optimization，近端策略优化
- 两个网络，分别是Actor和Critic


Flash-Attention

### MOE


## 多模态
Diffusion/GPT/CLIP/BLIP


## 分布式训练
DeepSpeed
- 选择 ZeRO Optimizer 的不同阶段。阶段0、1、2和3分别指禁用、优化器状态分区、优化器+梯度状态分区和优化器+梯度+参数分区。
Megatron

- limited GPU

use fp16 (this speeds up training)
use gradient_accumlation_steps (this simulates larger batch sizes)
use gradient_checkpointing (this uses disk to save RAM)
freeze model embeddings (this reduces weights to train)
freeze some model layers (this reduces weights to train)
use PEFT (this reduces weights to train)
increase LR and decrease epochs (this reduces work)
use smaller models (this reduces weights to train)

- 模型并行分为张量并行和流水线并行

## prompt


## RAG
- 主要针对大语言模型的幻觉、数据时效性、数据安全问题。

![](../.github/assets/02ml-llm-rag.png)


## Agents


## 问答
- 一个给定任务，如何优化LLM的效果
  - 从prompt engineering开始
  - RAG
  - fine tuning
- attention的加速优化
  - flash-attention 及 S2attention
- 如何扩展LLM的token
  - position embedding的角度
  - [Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2311.12351v1)
- 如何构建Instruction数据集
  - 预料生成，格式构造，提示词
- 如何处理训练中的loss spike
  - [adam在大模型预训练中的不稳定性分析及解决办法 - 丁晖的文章 - 知乎](https://zhuanlan.zhihu.com/p/675421518)
- 分布式训练
- 知识幻觉
  - 数据(数据重复、Bias、时效性， 一对多的映射关系)，训练（Imperfect representation learning、Parametric knowledge bias）
- 复读机问题/ 文本生成的重复问题
  - 多样性训练数据
  - 引入噪声
  - 温度参数调整
  - 后处理和过滤
- 灾难性遗忘
  - 重播缓冲区
  - 弹性权重共享
  - 增量学习
  - 多任务学习
  - 数据分布差异
  - 参数更新冲突
- Zero 应用的时候performance严重下降，为什么
- 怎么控制GAI不要给虚假答案
  - constitutional AI，red teaming 去帮助模型规范作答
  - fine tune一个小模型给specific task
- 文本生成的多跳问题
  - https://mp.weixin.qq.com/s/N0sjdNo-qWdZJ4UkXm-bdw
- 随着模型的增大，学习率越来越小。学习率与数据量、批量大小都没有明显的关系，且一般使用1e-3左右的学习率


## reference
- [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)
- A Survey of Large Language Models
- A Comprehensive Survey on Pretrained Foundation Models A History from BERT to ChatGPT
- [LLM推理优化技术综述：KVCache、PageAttention、FlashAttention、MQA、GQA](https://zhuanlan.zhihu.com/p/655325832)
- [The Rise and Potential of Large Language Model Based Agents: A Survey]()
- [SearchAnything](https://github.com/Immortalise/SearchAnything)
- [A Survey of Techniques for Maximizing LLM Performance](https://www.youtube.com/watch?v=ahnGLM-RC1Y)
- [https://github.com/NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [大模型检索增强生成（RAG）有哪些好用的技巧？ - Breezedeus的回答 - 知乎](https://www.zhihu.com/question/625481187/answer/3292724588)
- [FlashAttention 的速度优化原理是怎样的？](https://www.zhihu.com/question/611236756/answer/3310819022)
- [语言模型之Text embedding（思考篇） - 泽龙的文章 - 知乎](https://zhuanlan.zhihu.com/p/655310436)
- [大模型词表扩充必备工具SentencePiece - 吃果冻不吐果冻皮的文章 - 知乎](https://zhuanlan.zhihu.com/p/630696264)
- [十分钟读懂旋转编码（RoPE） - 绝密伏击的文章 - 知乎](https://zhuanlan.zhihu.com/p/647109286)
- [TH LLM Study Group 20231201](https://colab.research.google.com/drive/14Ls0gQktcuy4HYGQc0yY86X4sJ1lK2T8?usp=sharing#scrollTo=KMcHEPV-oAs0)
- [大语言模型是如何在预训练的过程中学习超长文本的呢？ - 段淇源的回答 - 知乎](https://www.zhihu.com/question/621810553/answer/3287188454)
- [大模型基础组件之位置编码-万字长文全面解读LLM中的位置编码与长度外推性（上） - OpenLLMAI的文章 - 知乎](https://zhuanlan.zhihu.com/p/626828066)
- [让LLM更好地学会中文：大模型继续预训练实践纪录 - Lil2J的文章 - 知乎](https://zhuanlan.zhihu.com/p/677653373)
- [如何解释大模型的重复生成现象？ - 慕谦或谦修的回答 - 知乎](https://www.zhihu.com/question/616130636/answer/3164017394)
