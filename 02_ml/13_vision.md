# 机器视觉

## 1. 传统视觉

- 均值滤波

## 2. 模型

### 2.1 分类

- ResNet
  - 解决网络过深带来的梯度消失问题
- ConvNext
- ViT
  - Transformer模型的视觉应用
- 深度可分离卷积

### 2.2 检测

参考[物体检测系统设计](../03_system/03_ml/object_detection.md)

- 一阶段检测
  - YOLO: 网格负责预测真实的bbox
  - SSD
  - RetinaNet
- 二阶段检测
  - rcnn, fast-rcnn, faster-rcnn
  - 特征抽取(feature extraction)，候选区域提取（Region proposal提取），边框回归（bounding box regression），分类（classification）
- 多阶段

  - Cascade-rcnn: 不同级采用不同 IoU 阈值来进行重新计算正负样本和采样策略来逐渐提高 bbox 质量

- anchor_base or anchor_free
- RPN
- 旋转目标检测
- NMS 非极大值抑制

### 2.3 分割

**语义分割**

- Unet

**实例分割**

### 2.4 生成

[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process)

> 图像生成相关：文本生成图像，图像生成图像，文本生成视频，文本生成语音。GAN、扩散模型、图像生成、多模态生成等。

**扩散模型**
存在一系列高斯噪声（T轮），将输入图片x0变为纯高斯噪声xt。模型则负责将xt复原回图片x0

![](../.github/assets/02ml-sd.png)

- autoencoder (VAE)
- U-Net
- text-encoder, CLIP Text Encoder

## 3. 代码

**IOU**

```python
import torch

def iou(box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # 左上角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2),   # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),   # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0   # 两个box没有重叠区域
    inter = wh[:,:,0] * wh[:,:,1]   # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # (N,)
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N,M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N,M)  # (N,M)

    iou = inter / (area1+area2-inter)
    return iou
```

```python
def compute_iou(boxA, boxB):
    # box:(x1,y1,x2,y2), x1,y1为左上角。原点为左上角，x朝右为正，y朝下为正。
    # 计算相交框的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交区域，并区域，及IOU。要和0比较大小，如果是负数就说明压根不相交
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

boxA = [1,1,3,3]
boxB = [2,2,4,4]
IOU = compute_iou(boxA, boxB)
```

- nms

```python
import numpy as np

def nms(boxes, scores, threshold):
    # boxes: 边界框列表，每个框是一个格式为 [x1, y1, x2, y2] 的列表
    # scores: 每个边界框的得分列表
    # threshold: NMS的IoU阈值

    # 按得分升序排列边界框
    sorted_indices = np.argsort(scores)
    boxes = [boxes[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    keep = []  # 保留的边界框的索引列表

    while boxes:
        # 取得分最高的边界框
        current_box = boxes.pop()
        current_score = scores.pop()

        keep.append(sorted_indices[-1])
        sorted_indices = sorted_indices[:-1]
        discard_indices = []  # 需要丢弃的边界框的索引列表

        for i, box in enumerate(boxes):
            # 计算与当前边界框的IoU
            iou = compute_iou(current_box, box)

            # 如果IoU超过阈值，标记该边界框为需要丢弃
            if iou > threshold:
                discard_indices.append(i)

        # 移除标记为需要丢弃的边界框。从后往前删，不然for循环会出错
        for i in sorted(discard_indices, reverse=True):
            boxes.pop(i)
            scores.pop(i)
            sorted_indices = np.delete(sorted_indices, i) # np与list的方法不同

    return keep
```

- DDPM

```python
import torch
import torch.nn as nn

class DenoiseDiffusion:
    def __init__(self, num_time_step=1000, device='cpu'):
        self.num_time_step = num_time_step
        self.device = device
        self.beta = torch.linspace(0.001, 0.02, num_time_step, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise=None):
        """Adds noise to the clean image x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))  # match shape for broadcasting
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def p_sample(self, model, x_t, t):
        """Performs a denoising step using the model prediction."""
        noise_pred = model(x_t, t)
        alpha_t = self.alpha[t].view(-1, *([1] * (x_t.dim() - 1)))
        alpha_bar_t = self.alpha_bar[t].view(-1, *([1] * (x_t.dim() - 1)))
        beta_t = self.beta[t].view(-1, *([1] * (x_t.dim() - 1)))

        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - coef * noise_pred)
        eps = torch.randn_like(x_t)
        return mean + torch.sqrt(beta_t) * eps

    def loss(self, model, x0):
        """Computes training loss."""
        B = x0.shape[0]
        t = torch.randint(0, self.num_time_step, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = model(x_t, t)
        return nn.functional.mse_loss(noise_pred, noise)
```

## 4. 问答

- 感受野
- 深度可分离卷积
- 数据增强
- diffusion model和stable diffusion公司的latent diffusion model特点
- Diffusion process
- 为什么diffusion model训练的时候需要1000 time steps，推理时只需要几十步
  - 训练采用的逻辑是基于DDPM的马尔可夫链逻辑，完整执行从t到t+1时刻的扩散过程；推理时采用的是DDIM类似的采样方法，将公式转化为非马尔可夫链的形式，求解任意两个时刻之间的对应公式，因此根据该公式可以在sample过程中跨步。

## 参考

- [Randaugment: Practical automated data augmentation with a reduced search space](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf)
- [关于Mixup方法的一个综述](https://zhuanlan.zhihu.com/p/439205252)
- [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
- [llava系列：llava、llava1.5、llava-next - AI狙击手的文章 - 知乎](https://zhuanlan.zhihu.com/p/695100288)
- https://github.com/DeepTecher/awesome-ChatGPT-resource-zh
- https://github.com/hua1995116/awesome-ai-painting
- https://www.zhihu.com/question/577079491/answer/2954363993
- https://www.zhihu.com/question/596230048
- [生成模型大道至简｜Rectified Flow基础概念｜代码 - 养生的控制人的文章 - 知乎](https://zhuanlan.zhihu.com/p/687740527)
- [How I Understand Diffusion Models](https://www.youtube.com/watch?v=i2qSxMVeVLI)
- [pytorch-stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion)
- [Diffusion学习路径记录（2023年） - Kylin的文章 - 知乎](https://zhuanlan.zhihu.com/p/605973097)
- [深入浅出完整解析Stable Diffusion（SD）核心基础知识 - Rocky Ding的文章 - 知乎](https://zhuanlan.zhihu.com/p/632809634)
