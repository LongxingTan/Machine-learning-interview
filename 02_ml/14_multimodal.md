# 多模态

多模态任务主要有Align和Fuse两种方式，或称为Light fusion和Heavy fusion。把不同模态的特征映射到相同的向量空间的操作叫align。代表作是OpenAI的CLIP，Google的Align。用一个模型把不同模态的特征混在一起用来完成某项任务，叫做Fuse。代表作有微软的UNITER，OSCAR等。

- Align 双塔
  - 向量内积，CLIP和ALIGN。
- Fusion 单塔
  - transformer, VLP, OSCAR, UNITER, VINFL


单流模型和双流模型
- 单流模型将图像侧和文本侧的embedding拼接到一起，输入到一个Transformer模型中。
- 双流模型让图像侧和文本侧使用两个独立的Transformer分别编码，可以在中间层加入两个模态之间的Attention来融合多模态信息


多模态对齐方法
- 借鉴BERT的思想进行masked language modeling对齐
- 使用contrastive loss进行多模态对齐


## 模型

VILBERT

CLIP

ALBEF
BLIP
- Q-Former的作用是什么

LLAVA


## 参考
- [大模型 | CLIP | BLIP | 损失函数代码实现 - 有点晕的文章 - 知乎](https://zhuanlan.zhihu.com/p/699507603)
