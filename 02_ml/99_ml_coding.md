# 机器学习面试代码

- [10min pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [60min pytorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Huggingface transformers course](https://huggingface.co/learn/nlp-course/en/chapter1/1)
- [ML code challenge](https://www.deep-ml.com/)

## 1. 目标与评价

**损失函数**

```python
import numpy as np

class MSE:
    def loss(self, y_true, y_pred):
        return 0.5 * np.power(y_true - y_pred, 2)

    def gradient(self, y_true, y_pred):
        # 损失函数对 y_pred (网络最后一层输出)的梯度
        return -1 * (y_true - y_pred)


class CrossEntropyLoss:
    """
    pytorch中labels的格式：nn.CrossEntropyLoss()是一维的类别, bce是one hot的多维
    ln(x)的导数 1/x，exp(x)的导数exp(x)
    """
    def loss(self, labels, logits, epsilon=1e-12):
        """
        labels = np.array([[0, 1], [1, 0]])
        logits = np.array([[0.1, 0.9], [0.8, 0.2]])
        """
        logits = np.clip(logits, epsilon, 1. - epsilon)
        return -np.mean(np.sum(labels * np.log(logits), axis=1))

    def gradient(self, labels, logits, epsilon=1e-12):
        logits = np.clip(logits, epsilon, 1. - epsilon)
        return -labels / logits


class CrossEntropyLoss2:
    def loss(self, labels, logits, epsilon=1e-12):
        """ 类似线形回归的shape
        labels = np.array([1, 0])
        logits = np.array([0.9, 0.2])
        """
        logits = np.clip(logits, epsilon, 1. - epsilon)
        return np.mean(- labels * np.log(logits) - (1 - labels) * np.log(1 - logits))

    def gradient(self, labels, logits, epsilon=1e-12):
        logits = np.clip(logits, epsilon, 1. - epsilon)
        return - (labels / logits) + (1 - labels) / (1 - logits)
```

- focal loss

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

**指标**

- AUC

```python

```

**cross validation**

```python

```

## 2. 统计学习模型

**线形回归: native**

**线性回归: numpy**

```python

```

**逻辑回归: native**

**逻辑回归: numpy**

```python

```

**决策树分类**

```python
import numpy as np

class TreeNode:
    def __init__(self):
        pass

class DecisionTree:
    def __init__(self):
        pass
```

**决策树回归**

**Xgboost回归**

**Xgboost分类**

**K-means**

```python
import numpy as np

class KMeansCluster:
    """
    examples: (n_example, n_feature)
    distance: (n_example, k)
    clusters: (n_example,)
    """
    def __init__(self, k, max_iter=100, tolerance=1e-4):
        self.k = k  # number of clusters
        self.max_iter = max_iter  # maximum number of iterations
        self.tolerance = tolerance  # convergence tolerance

    def fit(self, examples):
        n_examples, n_features = examples.shape
        centroids = examples[np.random.choice(n_examples, self.k, replace=False)]

        # Placeholder for storing cluster assignments
        clusters = np.zeros(n_examples)

        for _ in range(self.max_iter):
            # Step 1: Calculate distance between each example and centroids
            distances = np.zeros((n_examples, self.k))
            for i in range(self.k):
                distances[:, i] = self.euclidean_distance(examples, centroids[i])

            # Step 2: Assign each example to the closest centroid
            new_clusters = np.argmin(distances, axis=1)

            # Step 3: Check for convergence (if assignments haven't changed)
            if np.all(clusters == new_clusters):
                break

            clusters = new_clusters

            # Step 4: Update centroids by calculating the mean of the assigned points
            for i in range(self.k):
                centroids[i] = examples[clusters == i].mean(axis=0)

        self.centroids = centroids
        self.clusters = clusters

    def predict(self, examples):
        distances = np.zeros((examples.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = self.euclidean_distance(examples, self.centroids[i])
        return np.argmin(distances, axis=1)

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=-1))
```

**KNN**

**PCA**

```python
from numpy.linalg import svd

```

## 3. 深度学习模型

**MLP-numpy**

```python
import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(input_dim, output_dim) * 0.01  # (input_dim, output_dim)
        self.bias = np.zeros((1, output_dim))

    def forward(self, input):
        self.layer_input = input
        return np.matmul(input, self.weight) + self.bias

    def backward(self, accum_gradient, lr):
        # accum_gradient是loss对 layer_output的gradient, 形状相同
        grad_weight = np.matmul(self.layer_input.T, accum_gradient)  # (input_dim, output_dim)
        grad_bias = np.sum(accum_gradient, axis=0, keepdims=True)

        grad_input = np.matmul(accum_gradient, self.weight.T)  # (1, output_dim), weight更新前计算

        self.weight -= lr * grad_weight
        self.bias -= lr * grad_bias
        return grad_input
```

**MLP-torch**

**CNN-numpy**

- option1: native

```python
import numpy as np

class Conv2D:
    def __init__(self, kernel_size, input_channels, filters, padding, strides):
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.filters = filters
        self.padding_size = padding
        self.strides = strides
        # 标准正态分布的随机数，注意其形状
        self.kernels = np.random.randn(filters, input_channels, kernel_size, kernel_size)
        self.bias = np.zeros(filters)

    def forward(self, input):
        input_channels, h_in, w_in = input.shape
        assert input_channels == self.input_channels, "Input channels do not match the kernel input channels."

        # 填充输入数据。第一个轴（通常是通道轴）上不进行填充，第二个轴和第三个轴（通常是高度和宽度轴）上在开始和结束位置都填充padding个值
        input_pad = np.pad(input, ((0, 0), (self.padding_size, self.padding_size), (self.padding_size, self.padding_size)))
        self.layer_input = input_pad

        # h_out = (h_in + 2 * pad - k) // stride + 1
        h_out = (h_in + 2 * self.padding_size - self.kernel_size) // self.strides + 1
        w_out = (w_in + 2 * self.padding_size - self.kernel_size) // self.strides + 1
        output = np.zeros((self.filters, h_out, w_out))

        for f in range(self.filters):
            for i in range(h_out):
                for j in range(w_out):
                    i_start = i * self.strides
                    j_start = j * self.strides
                    im_region = input_pad[:, i_start:(i_start + self.kernel_size), j_start:(j_start + self.kernel_size)]
                    output[f, i, j] = np.sum(im_region * self.kernels[f]) + self.bias[f]
        return output

    def backward(self, accum_gradient, lr):
        # accum_gradient: the loss gradient for this layer's outputs
        input_gradient = np.zeros_like(self.layer_input)
        kernel_gradient = np.zeros_like(self.kernels)
        bias_gradient = np.zeros_like(self.bias)
        _, h_out, w_out = accum_gradient.shape

        for f in range(self.filters):
            for i in range(h_out):
                for j in range(w_out):
                    i_start = i * self.strides
                    j_start = j * self.strides
                    im_region = self.layer_input[:, i_start:(i_start + self.kernel_size), j_start:(j_start + self.kernel_size)]

                    kernel_gradient[f] += accum_gradient[f, i, j] * im_region
                    input_gradient[:, i_start:i_start + self.kernel_size, j_start:j_start + self.kernel_size] += (
                            accum_gradient[f, i, j] * self.kernels[f]
                    )

            bias_gradient[f] += np.sum(accum_gradient[f])

        self.kernels -= kernel_gradient * lr
        self.bias -= lr * bias_gradient

        if self.padding_size > 0:
            input_gradient = input_gradient[:, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]

        return input_gradient
```

- option2: 转化为一个大矩阵运算, 加快训练速度

```python
import numpy as np

def image2col():
    return

def col2image():
    return

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
            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()

    return output
```

**CNN-torch**

```python

```

```python
# https://github.com/openai/gpt-2/blob/master/src/model.py
import tensorflow as tf

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c
```

**LSTM-numpy**

```python

```

**LSTM-torch**

```python

```

**Attention-numpy**

```python

```

**Attention-torch**

```python
# https://nlp.seas.harvard.edu/annotated-transformer/

import torch
import torch.nn as nn

def scaled_dot_attention(q, k, v):
    d_k = k.size(-1)
    score = torch.matmul(q, k.transpose(-2, -1))
    score /= d_k ** 0.5
    score = torch.softmax(score, dim=-1)
    out = torch.matmul(score, v)
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q_state = self.q_proj(q)
        k_state = self.k_proj(k)
        v_state = self.v_proj(v)

        batch_size = q_state.size(0)
        # view只适合对满足连续性条件（contiguous）的tensor
        q_state = q_state.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_state = k_state.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_state = v_state.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        atten_output = scaled_dot_attention(q_state, k_state, v_state)
        atten_output = atten_output.transpose(1, 2).contiguous()  # view只前transpose或permute, 需要continuous
        atten_output = atten_output.view(batch_size, -1, self.d_k * self.num_heads)

        out = self.out_proj(atten_output)
        return out
```

**Dropout**

**BatchNorm**

**Activation**

## 4. 领域-NLP

**n-gram**

```python
# https://web.stanford.edu/~jurafsky/slp3/3.pdf

```

**tfidf**

[geeksforgeeks](https://www.geeksforgeeks.org/tf-idf-model-for-page-ranking/)

```python
# term-frequency: w represents a word, d means the document
# tf(w, d) = count(w, d) / total(d)

def compute_term_frequency(text: str, vocabularies: dict[str, int]) -> dict:
    """
    calculate term frequency: 每个document中，一个词出现次数越多越重要
    Args:
        text (str): input text
        vocabularies (dict[str, int]): vocabulary list from corpus

    Returns:
        dict: a dict containing the tf for each word
    """
    words = text.split(' ')
    word_count_norm = copy.deepcopy(vocabularies)
    for word in words:
        if word in word_count_norm.keys():
            word_count_norm[word] += 1
        else:
            # considering unknown words in testing
            word_count_norm["[UNK]"] += 1
    for word, count in word_count_norm.items():
        word_count_norm[word] = count / len(words)
    return word_count_norm  # 结果按vocab排序, 一个document可以根据tf转化为一个向量
```

```python
# Inverse Document Frequency: N is the total number of documents, while df(w) means the document frequency
# idf(w) = log(N / df(w))

def compute_inverse_document_frequency(documents: List[str]) -> dict[str, float]:
    """
    calculate the idf: 一个单词出现在越多document中，证明这个单词越不重要
    Args:
        documents (List[str]): a list of documents

    Returns:
        dict[str, float]: idf
    """
    N = len(documents)
    idf_dict = {}

    for document in documents:
        for word in set(document.split(' ')):
            # Count how many documents appear this word
            idf_dict[word] = idf_dict.get(word, 0) + 1

    # Apply logarithmic function to the counts
    idf_dict = {word: math.log(N / count) for word, count in idf_dict.items()}
    # Consider unknown words in the testing
    idf_dict['[UNK]'] = math.log(N + 1 / 1)
    return idf_dict
```

```python
def calculate_feature_vector(term_frequency: dict[str, int], inverse_document_frequency: dict[str, int]):
    tfidf = dict()
    for word, tf_word in term_frequency.items():
      tfidf[word] = tf_word * inverse_document_frequency[word]

    tfidf_vector = np.array([tfidf_word for _, tfidf_word in tfidf.items()])
    return tfidf_vector
```

**word2vec**

**Bayes文本分类器**

```python

```

**kv-cache**

**bert-summary**

**tokenizer: BPE贪心**

```python
# subword词表，之后编码和解码
import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()  # 使用空格进行区分
        for i in range(len(symbols)-1):  # 连续字符组成一个字符对
            pairs[symbols[i], symbols[i+1]] += freq  # 频率
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)  # 合并字符对
        v_out[w_out] = v_in[word]
    return v_out

# 设置待编码文本，key为字符，value为频率
# 每个单词后面增加</w>, 可以知道每个单词的结束位置, 而不会统计到下一个单词中
words = text.strip().split(" ")
word_freq_dict = collections.defaultdict(int)
for word in words:
    word_freq_dict[' '.join(word) + ' </w>'] += 1

vocab = {'l o w</w>' : 5, 'l o w e s t</w>' : 2, 'n e w e r</w>':6, 'w i d e r</w>':3}
num_merges = 10  # 迭代次数
for i in range(num_merges):
    pairs = get_stats(vocab)  # 字符字典
    best = max(pairs, key=pairs.get)  # 找到频率最高的字符对
    vocab = merge_vocab(best, vocab)
    print(best)
```

**positional encoding**

```python
import numpy as np
import torch

def get_positional_embedding(d_model, max_seq_len):
    positional_embedding = torch.tensor([
            [pos / np.power(10000, 2.0 * (i // 2) / d_model) for i in range(d_model)]  # i 的取值为 [0, d_model)
            for pos in range(max_seq_len)]  # pos 的取值为 [0, max_seq_len)
        )
    positional_embedding[:, 0::2] = torch.sin(positional_embedding[:, 0::2])
    positional_embedding[:, 1::2] = torch.cos(positional_embedding[:, 1::2])
    return positional_embedding
```

**beam search**

```python
# https://zhuanlan.zhihu.com/p/114669778

```

**top_k LLM token decoding**

```python
import numpy as np

def top_k_sampling(logits, k=5):
    """Perform top-k sampling on the logits array."""

    # Calculate the probabilities from logits
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    top_k_indices = np.argsort(probabilities)[-k:]

    # Normalize probabilities of top-k tokens
    top_k_probs = probabilities[top_k_indices] / np.sum(probabilities[top_k_indices])

    # Sample a token from the top-k tokens based on their probabilities
    sampled_token = np.random.choice(top_k_indices, p=top_k_probs)
    return sampled_token

logits = np.array([1.2, 0.8, 0.5, 2.0, 1.5])
sampled_token = top_k_sampling(logits, k=3)
print("Sampled token index:", sampled_token)
```

**prefix cache**
```python

```


## 5. 领域-CV

## 6. pipeline

**XGBoost**

**torch**

**PySpark**

## 7. 特征工程

**前处理-转换**

```python

```

**数量特征-pandas**

**数量特征-SQL**

**类别特征-pandas**

**类别特征-SQL**

## Reference

- [https://github.com/eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
