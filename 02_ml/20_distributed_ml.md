# 分布式机器学习

## 1. 原理

**GPU通信**

### 1.1 训练
**数据并行DP**
- 在每个计算节点上复制一份完整模型，将输入数据分成不同 batch 送入不同节点

**模型并行**
- 流水线并行PP
  - 适合模型层的大小相似的情况，如transformer
- 张量并行TP
  - 模型中的张量进行拆分分配到不同的 GPU 

**序列并行**
- Ulysses, Ring, TP-SP


### 1.2 推理
**量化 Quantization**

**剪枝 Pruning**

**蒸馏 Distillation**

**框架**
- TensorRT
- ONNX Runtime


## 2. 应用

### slurm 集群


### DP & parameter server
> 多用于单机多卡，一般采用参数服务器框架

[PyTorch 源码解读之 DP & DDP：模型并行和分布式训练解析 - OpenMMLab的文章 - 知乎](https://zhuanlan.zhihu.com/p/343951042)


### DDP
> 多用于多机多卡，采用Ring AllReduce通讯


### DeepSpeed
[https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
> 适用大模型sft

选择 ZeRO Optimizer 的不同阶段。阶段0、1、2和3分别指禁用、优化器状态分区、优化器+梯度状态分区和优化器+梯度+参数分区。

ZeRO：stage1 (optimizer state)，stage2 (+gradients)，stage3 (+model parameters)，offloads


### Megatron
[https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> 适用大模型pretrain


## 参考
**精读**
- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)

**扩展**
- [图解大模型训练系列之：DeepSpeed-Megatron MoE并行训练（原理篇） - 猛猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/681154742)
- [图解大模型训练之：数据并行上篇(DP, DDP与ZeRO) - 猛猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/617133971)
- [一文读懂「Parameter Server」的分布式机器学习训练原理 - 王喆的文章 - 知乎](https://zhuanlan.zhihu.com/p/82116922)
- [tensorflow ML框架外接ps方案 - 彭红卿的文章 - 知乎](https://zhuanlan.zhihu.com/p/396804900)
- [一文讲明白大模型分布式逻辑（从GPU通信原语到Megatron、Deepspeed） - 然荻的文章 - 知乎](https://zhuanlan.zhihu.com/p/721941928)
