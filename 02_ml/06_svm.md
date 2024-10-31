# 支持向量机SVM

间隔最大化来得到最优分离超平面。方法是将这个问题形式化为一个凸二次规划问题，还可以等价位一个正则化的合页损失最小化问题

## 1. 优化目标与间隔
目标: 支持向量离超平面的距离最远，其中支持向量：分类正确的样本中，距离边界最小的样本

hinge loss在训练后期天然偏重困难样本的损失，对于深度学习也有启发意义

KKT条件


## 2. 对偶


## 3. 核技巧


## 4. 软间隔


## 5. 问答

- what's the difference between logistic regression and SVM
  - loss type, logistic loss for LR, hinge loss for SVM
  - LR is parametric model (Bernoulli distribution), SVM with RBF kernel is non-parametric model
  - For SVM, only support vectors will influence the model, and every sample will influence LR model
  - SVM with structural risk minimization with L2 norm naturally, LR use experimential risk minimization
  - SVM normally use kernel function to solve the unlinear problem, LR not


## 6. 代码

```python
# Learning policy is to maximum the margin solved by the convex quadratic programming

import numpy as np
import random
from sklearn import datasets

def load_data():
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    return x, y

class SVM(object):
    def __init__(self, kernal, C):
        self.kernal = kernal
        self.C = C

    def fit(self, x_train, y_train):
        """by SMO"""
        self.x_train = x_train
        self.y_train = y_train
        self.N = x_train.shape[0]
        self.alpha = np.zeros((self.N, 1))  # alpha from the dual representation
        self.SMO(max_iter=1000)

    def predict_score(self, x_new):
        y_score = np.dot(x_new, self.w) + self.b
        return y_score[0]

    def predict(self, x_new):
        y_new = np.sign(self.predict(x_new))
        return y_new

    def SMO(self, max_iter):
        iter = 0
        while iter < max_iter:
            # step 1: choose two alpha
            iter += 1
            for i in range(self.N):
                j = random.randint(0, self.N - 1)
                if j == i:
                    continue
                else:
                    x_i, y_i, x_j, y_j = (
                        self.x_train[i, :],
                        self.y_train[i, 0],
                        self.x_train[j, :],
                        self.y_train[j, 0],
                    )
                    alpha_i, alpha_j = self.alpha[i, 0], self.alpha[j, 0]

                    L, H = self.calculate_LU(y_i, y_j, alpha_i, alpha_j)
                    self.calculate_w_b()

                    y_i_score = self.predict_score(x_i)
                    y_j_score = self.predict_score(x_j)

                    E_i = self.calculate_E(y_i_score, y_i)
                    E_j = self.calculate_E(y_j_score, y_j)

                    # step 2: update two alpha
                    alpha_j_new = alpha_j + y_j * (E_i - E_j) / self.calculate_gram(
                        x_i, x_j
                    )
                    alpha_j_new = min(alpha_j_new, H)
                    alpha_j_new = max(alpha_j_new, L)
                    alpha_i_new = alpha_i + y_i * y_j * (alpha_j - alpha_j_new)
                    print(alpha_j_new, alpha_i_new)
                    self.alpha[i, 0] = alpha_i_new
                    self.alpha[j, 0] = alpha_j_new

            self.w, self.b = self.calculate_w_b()
        return self.alpha

    def calculate_LU(self, yi, yj, alpha_i, alpha_j):
        if yi == yj:
            L = max(0, alpha_j + alpha_i - self.C)
            H = min(self.C, alpha_j + alpha_i)
        else:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        return L, H

    def calculate_E(self, y_hat, y):
        E = y_hat - y
        return E

    def calculate_gram(self, x_i, x_j):
        x_i = x_i.reshape(1, -1)
        x_j = x_j.reshape(1, -1)
        if self.kernal == "linear":
            k_ij = x_i.dot(x_i.T) + x_j.dot(x_j.T) - 2 * x_i.dot(x_j.T)
            return k_ij[0, 0]

    def calculate_w_b(self):
        self.w = np.dot(self.x_train.T, (self.alpha * self.y_train).reshape(-1, 1))
        self.b = np.mean(self.y_train - np.dot(self.x_train, self.w))
        return self.w, self.b

    def hinge_loss(self):
        pass

    def __str__(self):
        return "weights\t:%s\n bias\t:%f\n" % (self.w, self.b)

if __name__ == "__main__":
    x, y = load_data()
    y[np.where(y == 0)] = -1
    y[np.where(y == 2)] = -1
    y = y.reshape(-1, 1)
    print("x shape", x.shape)
    print("y shape", y.shape)
    svm = SVM(kernal="linear", C=1.0)
    svm.fit(x, y)
    y_hat = svm.predict(x)
```


## 参考
- [支持向量机（SVM）是什么意思](https://www.zhihu.com/question/21094489)
- [https://github.com/LasseRegin/SVM-w-SMO](https://github.com/LasseRegin/SVM-w-SMO)
- 