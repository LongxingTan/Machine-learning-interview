# 分布式机器学习

> 理解 how models are trained at scale, and deployed in production越来越重要

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

https://github.com/huggingface/picotron

## Trick

[使用pytorch，训练集数据太多达到上千万张，Dataloader加载很慢怎么办?](https://www.zhihu.com/question/356829360)
```python
import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
```


## 参考

**精读**

- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)
- [Tensor Parallelism with jax.pjit](https://irhum.github.io/blog/pjit/)

**扩展**

- [图解大模型训练系列之：DeepSpeed-Megatron MoE并行训练（原理篇） - 猛猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/681154742)
- [DDP系列第一篇：入门教程 - 996黄金一代的文章 - 知乎](https://zhuanlan.zhihu.com/p/178402798)
- [图解大模型训练之：数据并行上篇(DP, DDP与ZeRO) - 猛猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/617133971)
- [一文读懂「Parameter Server」的分布式机器学习训练原理 - 王喆的文章 - 知乎](https://zhuanlan.zhihu.com/p/82116922)
- [tensorflow ML框架外接ps方案 - 彭红卿的文章 - 知乎](https://zhuanlan.zhihu.com/p/396804900)
- [一文讲明白大模型分布式逻辑（从GPU通信原语到Megatron、Deepspeed） - 然荻的文章 - 知乎](https://zhuanlan.zhihu.com/p/721941928)
- [LLM实践--支线：分布式训练框架的编程基础 - 真中合欢的文章 - 知乎](https://zhuanlan.zhihu.com/p/10091011992)
- [torch.distributed](https://pytorch.org/docs/stable/distributed.html)
- [ML system 入坑指南 - Fazzie的文章 - 知乎](https://zhuanlan.zhihu.com/p/608318764)
