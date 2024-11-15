# 分布式机器学习

## 1. 原理
- 3D并行：数据并行DP、张量并行TP、流水线并行PP
- ZeRO：stage1 (optimizer state)，stage2 (+gradients)，stage3 (+model parameters)，offloads
- 模型并行分为张量并行和流水线并行

## 2. 应用

### DP和DDP


### parameter server


### DeepSpeed
选择 ZeRO Optimizer 的不同阶段。阶段0、1、2和3分别指禁用、优化器状态分区、优化器+梯度状态分区和优化器+梯度+参数分区。


### Megatron


## 参考
- [图解大模型训练系列之：DeepSpeed-Megatron MoE并行训练（原理篇） - 猛猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/681154742)
- [一文读懂「Parameter Server」的分布式机器学习训练原理 - 王喆的文章 - 知乎](https://zhuanlan.zhihu.com/p/82116922)
