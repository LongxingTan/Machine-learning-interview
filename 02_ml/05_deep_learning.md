# 深度学习

> 深入的应用参考[自然语言处理](./11_nlp.md)，[大语言模型](./12_llm.md)，[视觉](./13_vision.md)，[多模态](./14_multimodal.md)，[无监督/自监督](./08_unsuperwised.md)，[强化学习](./10_reinforcement.md)


## 1. 优化
### 1.1 前向后向传播 
[https://github.com/EurekaLabsAI/micrograd](https://github.com/EurekaLabsAI/micrograd)

- pytorch和jax的backprop
- 训练神经网络的一次迭代分三步：（1）前向传递计算损失函数；（2）后向传递计算梯度；（3）优化器更新模型参数
  - 前向传播，根据预测值和标签计算损失函数，以及损失函数对应的梯度。损失函数类的设计有正向值计算方法和梯度计算方法, 损失函数对y_hat的偏微分
  - 从loss梯度后向传播，计算每一个训练参数的grad。每一层后向传播的输入都是后面层的梯度。每一层有前向方法f(x)和后向方法f(grad)
  - 根据参数值和参数梯度进行优化更新参数: optimizer(w, w_grad)


### 1.2 优化 Optimizer
**梯度：**
- slope of a curve at a given point
- 从单变量看，抖的时候就走的步子大一点，缓的时候就走的小一点. 多个变量的不同变化决定了整体优化方向

**动量**
- 除了此刻的输入外，还考虑上一时刻的输出. 有的优化根据历史梯度计算一阶动量和二阶动量

**SGD原理**
- 考虑加入惯性，引入一阶动量，SGD with Momentum
- Very flexible—can use other loss functions
- Can be parallelized
- Slower—does not converge as quickly
- Harder to handle the unobserved entries (need to use negative sampling or gravity)

**Adam和adgrad区别和应用场景**
- Adam: 每个参数梯度增加了一阶动量（momentum）和二阶动量（variance），Adaptive + Momentum. 通过其来自适应控制步长，当梯度较小时，整体的学习率就会增加，反之会缩小

**RAdam**
- 用指数滑动平均去估计梯度每个分量的一阶矩(动量)和二阶矩(自适应学习率)，并用二阶矩去 normalize 一阶矩，得到每一步的更新量

**AdamW**
- 模型的优化方向是"历史动量"和"当前数据梯度"共同决定的

**对抗训练**
- 在训练过程中产生一些攻击样本，相当于是加了一层正则化，给神经网络的随机梯度优化限制了一个李普希茨的约束

**牛顿法**
- 梯度下降是用平面来逼近局部，牛顿法是用曲面逼近局部

**Batch Size**
- 用尽可能能塞进内存的batch size去train模型，提升训练速度. 但也存在trade-off
  - batch size过小，波动会比较大，不太容易收敛。但这种波峰，也有助于跳出局部最优，模型更容易有更好的泛化能力
  - batch size变大，步数整体变少，训练的步数更少，本来就波动就小，步数也少，同样本的情况下，你收敛的会更慢


```python
# example of gradient descent for a one-dimensional function
from numpy import asarray
from numpy.random import rand

def objective(x):
	return x**2.0

def derivative(x):
	return x * 2.0

def gradient_descent(objective, derivative, bounds, n_iter, step_size):
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

	for i in range(n_iter):
		gradient = derivative(solution)
		solution = solution - step_size * gradient
		solution_eval = objective(solution)
		print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

bounds = asarray([[-1.0, 1.0]])
n_iter = 30
step_size = 0.1
best, score = gradient_descent(objective, derivative, bounds, n_iter, step_size)
```

### 1.3 学习率scheduler
- LR与batch_size
  - 常用的heuristic 是 LR 应该与 batch size 的增长倍数的开方成正比，从而保证 variance 与梯度成比例的增长

```python
# Cyclic LR, 每隔一段时间重启学习率，这样在单位时间内能收敛到多个局部最小值，可以得到很多个模型做集成
scheduler = lambda x: ((LR_INIT-LR_MIN)/2)*(np.cos(PI*(np.mod(x-1,CYCLE)/(CYCLE)))+1)+LR_MIN

# warp up, 有助于减缓模型在初始阶段对mini-batch的提前过拟合现象，保持分布的平稳，同时有助于保持模型深层的稳定性
warmup_steps = int(batches_per_epoch * 5)
```

### 1.4 初始化

- 权重为什么不能被初始化为0?
  - 会导致激活后具有相同的值，网络相当于只有一个隐含层节点一样, hidden size失去意义


## 2. 损失函数
- MSE
  - prediction made by model trained with MSE loss is always normally distributed


- cross entropy/ 对数损失
  - `nn.CrossEntropyLoss(pred, label) = nn.NLLLoss(torch.log(nn.Softmax(pred)), label)`

$$ ce = - ylog(p) - (1-y)log(1-p) $$

- [binary cross entropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/)


- Focal loss
  - 对CE loss增加了一个调制系数来降低容易样本的权重值，使得训练过程更加关注困难样本。增加的这个系数就是评价难易，也就是概率的gamma次方
```python
import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, gamma, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds, targets):
        preds = preds.clamp(self.eps, 1 - self.eps)
        loss = (1 - preds) ** self.gamma * targets * torch.log(preds)  + preds ** self.gamma * (1 - targets) * torch.log(1 - preds)

        return -torch.mean(loss)
```


## 3. 网络模型结构

### 3.1 MLP

向量内积
- 表征两个向量的夹角，表征一个向量在另一个向量上的投影
- 表征加权平均

```python
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weight_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weight_error
        self.bias -= learning_rate * output_error
        return input_error
```

### 3.2 CNN

- Convolution is a mathematical operation trying to learn the values of filter(s) using backprop, where we have an input I, and an argument, kernel K to produce an output that expresses how the shape of one is modified by another.
- Convolutional layer is core building block of CNN, it helps with **feature detection.**
- Kernel K is a set of learnable filters and is small spatially compared to the image but extends through the full depth of the input image.
- **Dimension of the feature map** as a function of the input image size(W), feature detector size(F), Stride(S) and Zero Padding on image(P) is **(W−F+2P)/S+1**
- **No. of parameters** = (Kernel size * Kernel size * Dimension )+1 = 28
- 卷积等价于[一个大的矩阵一次性运算](Orthogonal Convolutional Neural Networks)
- CNN的 Inductive Bias(归纳偏置) 多过 vision transformer, CNN的归纳偏置，分别是 locality （局部性）和 translation equivariance（平移等变性）
- 在线卷积（Online Convolution）是在数据流式输入的情况下，实时计算卷积操作

```python
# https://github.com/openai/gpt-2/blob/master/src/model.py
def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c
```

```python
import numpy as np

def conv2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        for x in range(image.shape[0]):
            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum() + bias

    return output

def activation_fn(self, x):
    """A method of FFL which contains the operation and definition of given activation function."""
    if self.activation == 'relu':
        x[x < 0] = 0
        return x
    if self.activation == None or self.activation == "linear":
        return x
    if self.activation == 'tanh':
        return np.tanh(x)
    if self.activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    if self.activation == "softmax":
        x = x - np.max(x)
        s = np.exp(x)
        return s / np.sum(s)
```

```python
import numpy as np

def conv2d(inputs, kernels, bias, stride, padding):
    """ 正向卷积操作
    inputs: 输入数据，形状为 (C, H, W)
    kernels: 卷积核，形状为 (F, C, HH, WW)，C是图片输入层数，F是图片输出层数
    bias: 偏置，形状为 (F,)
    stride: 步长
    padding: 填充
    """
    # 获取输入数据和卷积核的形状
    C, H, W = inputs.shape
    F, _, HH, WW = kernels.shape

    # 对输入数据进行填充。在第一个轴（通常是通道轴）上不进行填充，在第二个轴和第三个轴（通常是高度和宽度轴）上在开始和结束位置都填充padding个值
    inputs_pad = np.pad(inputs, ((0, 0), (padding, padding), (padding, padding)))

    # 初始化输出数据，卷积后的图像size大小
    H_out = 1 + (H + 2 * padding - HH) // stride
    W_out = 1 + (W + 2 * padding - WW) // stride
    outputs = np.zeros((F, H_out, W_out))

    # 进行卷积操作
    for i in range(H_out):
        for j in range(W_out):  # 找到out图像对于的原始图像区域，然后对图像进行sum和bias
            inputs_slice = inputs_pad[:, i*stride:i*stride+HH, j*stride:j*stride+WW]
            # axis=(1, 2, 3)表示在通道、高度和宽度这三个轴上进行求和
            outputs[:, i, j] = np.sum(inputs_slice * kernels, axis=(1, 2, 3)) + bias            
    return outputs
```


### 3.3 RNN/LSTM/GRU

- 梯度爆炸与梯度消失
  - 梯度消失：在反向传播过程中累计梯度一直相乘，当很多小于1的梯度出现时导致前面的梯度很小，难以学习long-term dependencies
    - 一般改进: 改进模型
  - 梯度爆炸：the exploding gradient problem当梯度较大，链式法则导致连乘过大,数值不稳定
    - 一般改进: 梯度截断, 权重衰减
  - 通过多个gate
- 长距离依赖问题
- 计算复杂度：
  - LSTM: 序列长度 x（hidden**2）
- RNN的inductive bias是sequentiality和time invariance，即序列顺序上的time-steps有联系，和**时间变换的不变性**（rnn权重共享）


### 3.4 Transformer
- 结构
  - encoder: embed + layer(self-attention, skip-connect, ln, ffn, skip-connect, ln) * 6
  - decoder: embed + layer(self-attention, cross-attention, ffn, skip-connect, ln) * 6


- attention
  - $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
  - 通过使用key向量，模型可以学习到不同模块之间的相似性和差异性，即对于不同的query向量，它可以通过计算query向量与key向量之间的相似度，来确定哪些key向量与该query向量最相似。
  - kq的计算结果，形成一个（n，n）邻接矩阵，再与v相乘形成加权平均的消息传递
  - MLP-mixer提出的抽象，attention是token-mixing，ffn是channel-mixing
  - 多头注意力？增强网络的容量和表达能力，类比CNN中的不同channel
  - 时间和空间复杂度
    - sequence length n, vector representations d. QK矩阵相乘复杂度为O(n^2 d), softmax与V相乘复杂度O(n^2 d)
    - FFN复杂度：O(n d^2)
  - 优化：kv-cache，MQA，GQA
    - kv cache: 空间换时间，自回归中每次生成一个token，前面的token计算存在重复性
    - Multi Query Attention: MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量
    - Group Query Attention: 将查询头分成N组，每个组共享一个Key 和 Value 矩阵
    - Flash attention: 利用GPU硬件非均匀的存储器层次结构实现内存节省和推理加速

- attention为什么除以根号d
  - 称为attention的temperature。如果输入向量的维度d比较大，那么内积的结果也可能非常大，这会导致注意力分数也变得非常大，可能会使得softmax函数的计算变得不稳定(接近one-hot, 梯度消失)，并且会影响模型的训练和推理效果。通过除以根号d，可以将注意力分数缩小到一个合适的范围内，从而使softmax函数计算更加稳定，并且更容易收敛。
  - Google T5采用Xavier初始化缓解梯度消失，从而不需要除根号d

- 使用正弦作为位置编码
  - self-attention无法表达位置信息，由位置编码提供位置信息
  - 绝对位置编码的案例（sinusoidal、learned）、相对位置编码的案例（T5、XLNet、DeBERTa、ALiBi等）、旋转位置编码（RoPE、xPos）

- Positional Encoding/Embedding 区别
  - 学习式(learned)：直接将位置编码当作可训练参数，比如最大长度为 512，编码维度为 768，那么就初始化一个 512×768 的矩阵作为位置向量，让它随着训练过程更新。BERT、GPT 等模型所用的就是这种位置编码
  - 固定式(fixed)：位置编码通过三角函数公式计算得出
    - $$ \begin{aligned} PE_{(pos,2i)} & = sin(pos/10000^{2i/d_{model}})\ \end{aligned} $$
    - $$ \begin{aligned} PE_{(pos,2i+1)} & = cos(pos/10000^{2i/d_{model}}) \ \end{aligned} $$

- masking
  - Q*K结果上，加一个很大的负数，或乘？

- LSTM相比Transformer有什么优势

- attention瓶颈
  - low rank，talking-head

- Transformer是如何处理可变长度数据的？
  - 可变长度的意思: 模型训练好了，一个新的序列长度样本也可以作为输入. 但一个batch内仍需要padding到同一长度
  - 只需要保持参数矩阵维度与输入序列的长度无关，例如全连接层针对feature, 都不影响sequence维度; attention等也都是

- warmup预热学习率
  - 在训练开始时，模型的参数初始值是随机的，模型还没有学到有效的特征表示。如果此时直接使用较大的学习率进行训练，可能会导致模型的参数值更新过快，从而影响模型的稳定性和收敛速度。此时使用warmup预热学习率的策略可以逐渐增加学习率，使得模型参数逐渐收敛到一定的范围内，提高模型的稳定性和收敛速度。

- KV Cache
  - 加速推断, 解码过程是一个token一个token生成，如果每一次解码都从输入开始拼接好解码的token，那么会有非常多的重复计算
  - 矩阵乘法性质: 矩阵可以分块，将矩阵A拆分为[:s], [s]两部分，分别和矩阵B相乘，那么最终结果可以直接拼接

```python
def scaled_dot_product(q, k, v, softmax, attention_mask, attention_dropout):   
    outputs = tf.matmul(q, k, transpose_b=True)
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    outputs = outputs / dk
    # if attention_mask is not None:
    #     outputs = outputs + (1 - attention_mask) * -1e9

    outputs = softmax(outputs, mask=attention_mask)
    outputs = Dropout(rate=attention_dropout)(outputs)
    outputs = tf.matmul(outputs, v)  # shape: (m,Tx,depth), same shape as q,k,v
    return outputs


# multi-head有多种写法: 变成4维的 (batch_size, -1, num_heads, d_k), 变成3维的(batch * num_heads, -1, d_k), 以及下面的循环
class FullAttention(tf.keras.layers.Layer):
    def __init__(self,d_model, num_of_heads, dropout, d_out=None):
        super().__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.dropout = dropout
        self.depth = d_model // num_of_heads
        self.wq = [Dense(self.depth//2, use_bias=False) for i in range(num_of_heads)]
        self.wk = [Dense(self.depth//2, use_bias=False) for i in range(num_of_heads)]
        self.wv = [Dense(self.depth//2, use_bias=False) for i in range(num_of_heads)]
        self.wo = Dense(d_model if d_out is None else d_out, use_bias=False)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, q, k, v, attention_mask=None, training=False):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](q)
            K = self.wk[i](k)
            V = self.wv[i](v)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax, attention_mask, self.dropout))

        multi_attn = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_attn)

        return multi_head_attention
```

### 3.5 正则化
- 对模型施加显式的正则化约束
  - L1/L2 weight decay
  - dropout,
  - batch normalization，residual learning, label smoothing
- 利用数据增广的方法，通过数据层面对模型施加隐式正则化约束


```python
# 标签平滑，hard label转变成soft label，使网络优化更加平滑。有效正则化工具，通过在均匀分布和hard标签之间应用加权平均值来生成soft标签。用于减少训练的过拟合问题并进一步提高分类性能
targets = (1 - label_smooth) * targets + label_smooth / num_classes
```

### 3.6 Norm
**Batch Norm**
- BN用来减少 “Internal Covariate Shift” 来加速网络的训练，BN 和 ResNet 的作用类似，都使得 loss landscape 变得更加光滑了 (How Does Batch Normalization Help Optimization)
- BN在训练和测试过程中，其均值和方差的计算方式是不同的。测试过程中采用的是基于训练时估计的统计值，训练过程中则是采用指数加权平均计算
- 注意有可训练的参数scale和bias
- BN，当 batch 较小时不具备统计意义，而加大的 batch 又受硬件的影响；BN 适用于 DNN、CNN 之类固定深度的神经网络，而对于 RNN 这类 sequence 长度不一致的神经网络来说，会出现 sequence 长度不同的情况
- 分布式训练时，BN的跨卡通信
  - [Implementing Synchronized Multi-GPU Batch Normalization](https://hangzhang.org/PyTorch-Encoding/tutorials/syncbn.html)

**Layer Norm**
- layer normalization 有助于得到一个球体空间中符合0均值1方差高斯分布的 embedding， batch normalization不具备这个功能
- LayerNorm可以对输入进行归一化，使得每个神经元的输入具有相似的分布特征，从而有助于网络的训练和泛化性能。此外，由于归一化的系数是可学习的，网络可以根据输入数据的特点自适应地学习到合适的归一化系数。
- 加速模型的训练。由于输入已经被归一化，不同特征之间的尺度差异较小，因此优化过程更容易收敛，加快了模型的训练速度。
- 为什么不用batch norm? BN广泛用于CV，针对同一特征、跨样本开展归一。样本之间仍然具有可比较性，但特征与特征之间不再具有可比较性。NLP中关键的不在于样本中同一特征的可比较
- 由于BN需要统计不同样本统计值，因此分布式训练需要sync BatchNorm, Layer Norm则不需要

```python
# layer norm: https://www.kaggle.com/code/cpmpml/graph-transfomer?scriptVersionId=24171638&cellId=18
mean = K.mean(inputs, axis=-1, keepdims=True)
variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
std = K.sqrt(variance + self.epsilon)
outputs = (inputs - mean) / std
if self.scale:
    outputs *= self.gamma
if self.center:
    outputs += self.beta
```

```python
def GroupNorm(x, gamma, beta, G, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    return x * gamma + beta
```

[RMSNorm - Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467.pdf)
- RMSNorm舍弃了中心化操作(re-centering)，归一化过程只实现缩放(re-scaling)，缩放系数是均方根(RMS)


### 3.7 pool

```python
def get_pools(img: np.array, pool_size: int, stride: int) -> np.array:
    pools = []

    # Iterate over all row blocks (single block has `stride` rows)
    for i in np.arange(img.shape[0], step=stride):
        # Iterate over all column blocks (single block has `stride` columns)
        for j in np.arange(img.shape[0], step=stride):
            # Extract the current pool
            mat = img[i:i+pool_size, j:j+pool_size]
            # Make sure it's rectangular - has the shape identical to the pool size
            if mat.shape == (pool_size, pool_size):
                # Append to the list of pools
                pools.append(mat)
    return np.array(pools)

def max_pooling(pools: np.array) -> np.array:    
    num_pools = pools.shape[0]  # Total number of pools
    # Shape of the matrix after pooling - Square root of the number of pools
    tgt_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))

    pooled = []
    for pool in pools:
        pooled.append(np.max(pool))
    return np.array(pooled).reshape(tgt_shape)
```

### 3.8 dropout
- 训练时，根据binomial分布随机将一些节点置为0，概率为p，剩神经元通过乘一个系数(1/(1-p))保持该层的均值和方差不变；预测时不丢弃神经元，所有神经元输出会被乘以(1-p)
- 参考AlphaDropout，普通dropout+selu激活函数会导致在回归问题中出现偏差


## reference
- [小白都能看懂的超详细Attention机制详解 - 雅正冲蛋的文章 - 知乎](https://zhuanlan.zhihu.com/p/380892265)
- [https://github.com/tmheo/deep_learning_study](https://github.com/tmheo/deep_learning_study)
- [https://zybuluo.com/hanbingtao/note/581764](https://zybuluo.com/hanbingtao/note/581764)
- [Bert/Transformer 被忽视的细节（或许可以用来做面试题） - LiteAI的文章 - 知乎](https://zhuanlan.zhihu.com/p/613407791)
- [深度网络loss除以10和学习率除以10是不是等价的？ - 走遍山水路的回答 - 知乎](https://www.zhihu.com/question/320377013/answer/2591409899)
- [为什么Layer Norm反向传播的梯度会接近零？ - JoJoJoJoya的回答 - 知乎](https://www.zhihu.com/question/570354498/answer/2788826325)
- [LSTM如何来避免梯度弥散和梯度爆炸？ - Quokka的回答 - 知乎](https://www.zhihu.com/question/34878706/answer/665429718)
- [NLP中的Transformer架构在训练和测试时是如何做到decoder的并行化的？ - 市井小民的回答 - 知乎](https://www.zhihu.com/question/307197229/answer/1859981235)
- [碎碎念：Transformer的细枝末节 - 小莲子的文章 - 知乎](https://zhuanlan.zhihu.com/p/60821628)
- [优化时该用SGD，还是用Adam？](https://blog.csdn.net/S20144144/article/details/103417502)
- [对数损失函数](https://www.zhihu.com/question/27126057)
- [关于Mixup方法的一个综述](https://zhuanlan.zhihu.com/p/439205252)
- https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-pytorch-loss-functions.md
- https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling
- [KV cache详解 图示，显存，计算量分析](https://zhuanlan.zhihu.com/p/646577898)
- [详解深度学习中的梯度消失、爆炸原因及其解决方法 - DoubleV的文章 - 知乎](https://zhuanlan.zhihu.com/p/33006526)
- [浅谈后向传递的计算量大约是前向传递的两倍 - 回旋托马斯x的文章 - 知乎](https://zhuanlan.zhihu.com/p/675517271)
- [从 0 手撸一个 pytorch - 易迟的文章 - 知乎](https://zhuanlan.zhihu.com/p/675673150)
- [如何理解Adam算法(Adaptive Moment Estimation)？ - Summer Clover的回答 - 知乎](https://www.zhihu.com/question/323747423/answer/2576604040)
- [五、参数量、计算量FLOPS推导 - 小明的HZ的文章 - 知乎](https://zhuanlan.zhihu.com/p/676113501)
- [史上最细节的自然语言处理NLP/Transformer/BERT/Attention面试问题与答案 - 海晨威的文章 - 知乎](https://zhuanlan.zhihu.com/p/348373259)
- [Transformer学习笔记一：Positional Encoding（位置编码） - 猛猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/454482273)
- [PyTorch 源码解读系列 - OpenMMLab的文章 - 知乎](https://zhuanlan.zhihu.com/p/328674159)
- [对比pytorch中的BatchNorm和LayerNorm层 - 严昕的文章 - 知乎](https://zhuanlan.zhihu.com/p/656647661)
- [万字综述，核心开发者全面解读PyTorch内部机制](https://mp.weixin.qq.com/s/8J-vsOukt7xwWQFtwnSnWw)
- [一文搞懂混合精度训练原理 (常用O1) - APlayBoy的文章 - 知乎](https://zhuanlan.zhihu.com/p/701452410)
- [一文讲明白大模型分布式逻辑（从GPU通信原语到Megatron、Deepspeed） - 然荻的文章 - 知乎](https://zhuanlan.zhihu.com/p/721941928)
- [深度学习中，是否应该打破正负样本1:1的迷信思想？ - 密排六方橘子的回答 - 知乎](https://www.zhihu.com/question/654186093/answer/3483543427)
