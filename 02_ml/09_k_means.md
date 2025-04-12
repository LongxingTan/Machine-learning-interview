# k-means

partitioning a dataset into k distinct clusters based on similarity measures. It aims to minimize the within-cluster sum of squares (WCSS) or the average squared distance between data points and their assigned cluster centroids
通过样本间的相似性对数据集进行聚类，使类内差距最小化，类间差距最大化

## 过程

- 选择初始化的 k 个样本作为初始聚类中心
- 针对数据集中每个样本, 计算它到 k 个聚类中心的距离并将其分到距离最小的聚类中心所对应的类中
- 针对每个类别 ，重新计算它的聚类中心 （即属于该类的所有样本的质心）；
- 重复上面 2 3 两步操作，直到达到某个中止条件（迭代次数、最小误差变化等

## 评价

- 可以通过衡量簇内差异来衡量聚类的效果：Inertia
  - a.它的计算太容易受到特征数目的影响。
  - b.它不是有界的，Inertia是越小越好，但并不知道何时达到模型的极限，能否继续提高。
  - c.它会受到超参数K的影响，随着K越大，Inertia必定会越来越小，但并不代表模型效果越来越好。
  - d.Inertia 对数据的分布有假设，它假设数据满足凸分布，并且它假设数据是各向同性的，所以使用Inertia作为评估指标，会让聚类算法在一些细长簇、环形簇或者不规则形状的流形时表现不佳。
- 轮廓系数

## PCA

- explained variance ratio
- create new uncorrelated variables that successively maximize variance by solving an eigenvalue/eigenvector problem.
- reduce the dimensionality of dataset, increase interpretability while minimize information loss
- pros: no need of prior; reduce overfitting (by reduce #variables in the dataset); visualizable
- cons: data standardization is a prerequisite; information loss

## 问答

- cons

  - 对outlier敏感

- k means 如何选择k

  - scree plot: 横坐标n_cluster, 纵坐标intra-cluster variance (区分 inter-cluster variance)
  - [Stop using the elbow criterion for k-means](https://arxiv.org/pdf/2212.12189)

- 怎么判断clustering效果好不好

  - 聚类评价指标: Purity, NMI, RI, Precision(查准率), Recall(查全率), F, ARI, Accuracy(正确率)

- k-means和KNN的区别

  - k-means无监督，KNN有监督

- signal-to-variance ratio

- K-means为什么是收敛的
- K-means 怎么初始化 K-means++
- EM方法为什么是收敛的

- Any acceleration algorithm for PCA
  - PCA involves computing the eigenvectors and eigenvalues of the covariance matrix or performing Singular Value Decomposition (SVD), both of which can be time-intensive.

## code

```python
# https://gancode.com/2021/03/01/6933952373303803912.html
import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters=3, random_state=0):
        assert n_clusters >=1, " must be valid"
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._center = None  # cluster中心, n_cluster * n_feature
        self.cluster_centers_ = None

    def distance(self, M, N):
        return (np.sum((M - N) ** 2, axis = 1))** 0.5

    def _generate_labels(self, center, X):
        return np.array([np.argmin(self.distance(center, item)) for item in X])

    def _generate_centers(self, labels, X):
        return np.array([np.average(X[labels == i], axis=0) for i in np.arange(self._n_clusters)])

    def fit_predict(self, X, n_iters=1000, tol=1e-4):
        # X: 样本, n_sample * n_feature
        k = self._n_clusters

        # 设置随机数
        if self._random_state:
            random.seed(self._random_state)

        # 生成随机中心点的索引
        center_index = [random.randint(0, X.shape[0]) for _ in np.arange(k)]
        center = X[center_index]

        while n_iters > 0:
            # 记录上一个迭代的中心点坐标
            last_center = center

            # 根据上一批中心点，计算各个点所属的类
            labels = self._generate_labels(last_center, X)
            self.labels_ = labels

            # 新的中心点坐标
            center = self._generate_centers(labels, X)
            # if np.linalg.norm(center - self.cluster_centers_) < tol:
            #     break
            self.cluster_centers_ = center

            # 如果新计算得到的中心点，和上一次计算得到的点相同，说明迭代已经稳定了。
            if (last_center == center).all():
                self.labels_ = self._generate_labels(center, X)
                break

            n_iters = n_iters - 1
        return self
```

时间复杂度： O(tkmn) ，t 为迭代次数，k 为簇的数目，n 为样本点数，m 为样本点维度 <br>
空间复杂度： O(m(n+k)) ，k 为簇的数目，m 为样本点维度，n 为样本点数

```python
# PCA: np.linalg.norm

```

## 参考

- [K-Means Clustering](https://towardsdatascience.com/k-means-clustering-8e1e64c1561c)
- [一文读懂K均值（K-Means）聚类算法](https://mp.weixin.qq.com/s/MsmelZvW8p7mJ2O9JWOm1g)
- [【机器学习】K-means](https://zhuanlan.zhihu.com/p/78798251)
- [根因分析初探：一种报警聚类算法在业务系统的落地实施](https://tech.meituan.com/2019/02/28/root-clause-analysis.html)
- [聚类评价指标](https://zhuanlan.zhihu.com/p/53840697)
- [高斯混合模型（GMM）详解 - TroubleShooter的文章 - 知乎](https://zhuanlan.zhihu.com/p/655018030)
