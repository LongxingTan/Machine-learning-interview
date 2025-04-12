# KNN

**应用场景**

- 产品推荐和推荐引擎
- 自然语言处理 (NLP) 算法的相关性排名
- 图像或视频的相似性搜索

**过程**

- 计算训练样本和测试样本中每个样本之间的距离
- 对距离进行排序
- 选择距离最小的k个样本: K小，噪音多，K大，边界模糊
- 根据k个样本的标签进行投票, 得到分类类别

**距离**

- 欧几里得距离
- 明可夫斯基距离
- 曼哈顿距离
- 余弦相似度

**优化**

- NDTree
- Approximate kNN 加速服务
  - Tree-based ANN
  - Locality-sensitive hashing (LSH)
  - Clustering based
- faiss

## 问答

**Pros**:

- Almost no assumption on f other than smoothness
- High capacity/complexity
- High accuracy given a large training set
- Supports online training (by simply memorizing)
- Readily extensible to multi-class and regression problems

**Cons**:

- Storage demanding
- Sensitive to outliers
- Sensitive to irrelevant data features (vs. decision trees)
- Needs to deal with missing data (e.g., special distances)
- Computationally expensive: O(ND) time for making each prediction
- Can speed up with index and/or approximation

## code

```python
# https://towardsdatascience.com/create-your-own-k-nearest-neighbors-algorithm-in-python-eb7093fc6339
# 考虑 max heap

import numpy as np

def eu_dist(x_train, x_test):
    distances = []
    for x_cur in x_train:
        distance = np.sqrt(np.sum((x_cur - x_test) ** 2))
        distances.append(distance)
    return distances

def KNN(x_train, y_train, x_test, k):
    # convert to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)

    # calculate the distance between x_train and x_test
    distances = eu_dist(x_train, x_test)
    # sort the distance and find the top-k in y_train
    dist_index = np.argsort(distances)[:k]
    topk_y = [y_train[index] for index in dist_index]

    # count the frequency in y_train
    dict = {}
    for y_cur in topk_y:
        y_cur = y_cur[0]
        dict[y_cur] = dict.get(y_cur, 0) + 1

    dict = list(dict.items())
    dict = sorted(dict, key=lambda x: x[1], reverse= True)

    # return the highest frequent value
    predict = dict[0][0]
    return predict
```

## Reference

- [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
