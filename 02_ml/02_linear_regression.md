# 线性回归 <!-- omit in toc -->

- [0. 基础假设](#0-基础假设)
- [1. 最小二乘法 Ordinary least squares](#1-最小二乘法-ordinary-least-squares)
- [2. 最大似然法 Maximum likelihood estimation](#2-最大似然法-maximum-likelihood-estimation)
- [3. 最大后验法与正则 Maximum a posteriori estimation](#3-最大后验法与正则-maximum-a-posteriori-estimation)
  - [L1正则](#l1正则)
  - [L2正则](#l2正则)
- [4. 贝叶斯线性回归与高斯过程回归](#4-贝叶斯线性回归与高斯过程回归)
- [5. 问答](#5-问答)
- [6. 代码](#6-代码)
- [参考](#参考)

## 0. 基础假设

**Linearity**: There is a linear relationship between the independent variables(X) and the dependent variables (y)
  - ✅ Tip: Check with scatter plots or residual vs fitted plots.

**Independence of Errors**: Independence assumes that there is no relationship or correlation between the errors (residuals) of different observations.
  - ✅ Tip: Use Durbin-Watson test for autocorrelation.

**Normality of Errors**: The residuals of the linear regression model are assumed to be normally distributed.
  - ✅ Tip: Use Q-Q plots or the Shapiro-Wilk test.

**Homoscedasticity** (Constant Variance of Errors): Homoscedasticity assumes that the variability of the errors (residuals) is constant across all levels of the independent variables.
  - ✅ Tip: Check with a residual plot (should look like random scatter).

**No Multicollinearity**: No Multi-collinearity between features
  - ✅ Tip: Check Variance Inflation Factor (VIF).
    VIF > 5 (or 10) suggests problematic multicollinearity.

**IID (Independent and Identically Distributed)**: an underlying assumption of linear regression — but more specifically, it's an assumption about the error terms (residuals), not the raw data itself.
  - Residuals are not correlated with each other.
  - Residuals have the same variance.
  - (Often) Residuals are normally distributed (not strictly required for prediction, but essential for inference like p-values or confidence intervals).
- ✅ Tip: IID Assumption Checks

  | Check                       | Tool/Test                  | Good Sign                   |
  |----------------------------|----------------------------|-----------------------------|
  | **Independence**           | Durbin-Watson              | Value ≈ 2                   |
  | **Identical distribution** | Residual vs Fitted plot, Breusch-Pagan test | Random scatter, p > 0.05    |
  | **Normality**              | Q-Q plot, Shapiro-Wilk test| Diagonal line, p > 0.05     |


## 1. 最小二乘法 Ordinary least squares

> 最小二乘法对线性回归的基础假设是：误差（residual）符合正态分布。

最小二乘法的理念是承认观测误差的存在，误差是围绕真值上下波动的，从几何角度看，当平方误差最小时即为真值。

具体应用时，算法目标是求线性方程中未知权重w和b，经验风险最小化时的权重认为是最优的。为了简化，我们经常在推导中忽略b，因为b可以认为存在一个常数列x=1对应的权重w的一个分量，不必单独另求。

以下向量定义为列向量，注意不同步骤需要清楚定义**谁被看作是谁的函数**

$$
y=w^Tx +b
$$

这里，有一个转置T，是为了求两个向量的点积。

$$
w=\mathop {argmin} _{w}Loss(w)
$$

Loss function:

$$
L(w)=\frac{1}{n}\sum_{i=1}^{n}{(y-\hat{y})^2}
$$

从标准量中计算其梯度(只有一个样本，多样本加连加符号):

$$
\frac{\partial{L(w)}}{\partial{w_j}} = {(\hat{y} - y)}{x_j}
$$

梯度是让损失函数下降的方向。先看单个参数，就决定了下一步的大小(学习率还会拦一道)和方向(也就是正负)。对多个特征需要的多参数，主要是决定了不同参数大小的对比，也就成了多向量空间中的方向。对于多个样本来说，可以认为每个样本都会让参数"优化"移动一次。
不同参数的移动大小，可以说决定了这个参数的重要程度。因为本身需要norm，让这个移动更纯粹一些。

其矩阵形式如下:

$$
L= (y-XW)^{T}(y-XW)
$$

$$
L=y^Ty-y^TXW-W^TX^Ty+W^TX^TXW
$$

继续化简需要用到矩阵求导的几个规律：

$$
\frac{\partial{L}}{\partial{w}}=2X^TXw-2X^Ty
$$

- 损失函数为凸函数，可知道全局最优解的取值为导数等于0处的极值
- 解析解的存在需要满足满秩矩阵。加L2 norm之后有了先验约束，则不需要
- 梯度下降的优化方法：梯度决定了参数的变化方向。模型中每一个参数都看做是损失函数的自变量。结合逻辑回归、深度学习的前向反向传播综合理解

Normal Equation: 解析解-> 导数为0:

$$
W=(X^TX)^{-1}X^Ty
$$

## 2. 最大似然法 Maximum likelihood estimation

最小二乘与最大似然等价，同属于频率派(Frequentists)；最大后验则属于贝叶斯派(Bayesians)

什么是似然(likelihood)？在似然中，所有数据对(x, y) 的集合，被看成是一个随机变量X，但X已经被观察到了，已经固定。以最常见的抛硬币模型举例，硬币出现正面或反面就是一个随机变量，一个正常硬币出现正面的概率为1/2。而似然的定义是给定一个硬币，我们抛了10次之后出现了1次正面，出现1/10正面这个事件已经被观察到了并固定，这个硬币是正常硬币的可能性就被称为似然。

$$
w,b=argmax(logP(x|w))
$$

在线性回归中，贝叶斯公式可以写成:

$$
P(w|X)=\frac{P(X|W).P(w)}{P(X)}
$$

- P(X|w)是似然函数，
- P(w)被称为先验(prior)，是发生在我们观察样本之前，例如硬币抛出之前对硬币的假设。
- P(w|X)就称为后验概率(posterior)

最大似然：

$$
w_{MLE}=\mathop{argmax}_w P(x|w)
$$

最大似然需要假设分布，对于一般线性回归的假设即是: 误差是一个高斯分布。(注意区分后验概率中的w本身是一个高斯分布)

关于变量x的高斯分布公式：

$$
P(x)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$

误差满足高斯分布，而误差是真值和预测的差值，所以：

$$
P(\hat{y}-xw) \sim N(0，\sigma^2)
$$

由于误差是均值为0，方差$\sigma^2$的高斯分布，则y的预测值是满足均值为wx，方差为$\sigma^2$的高斯分布。在似然中的x其实代表的是一个样本，即（x,y)数据对，而在特征已知的情况下，似然函数可以进一步表示为：

$$
P(\hat{y}|x,w)
$$

因此，似然也可以表达成一个高斯分布

$$
P(y|x,w) \sim N(w^Tx,\sigma^2)
$$

可以转化为：

$$
P(y|x,w)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y-w^Tx)^2}{2\sigma^2})
$$

## 3. 最大后验法与正则 Maximum a posteriori estimation

贝叶斯认为变量w也是一个随机变量，随机变量自带一个概率分布。因此在观察数据之前，我们可以提前根据经验预估一个w的分布，这个预估就是先验，并成为一个对参数的约束，即为最大后验法。

$$
w_{MAP}=\mathop{argmax}_w P(x|w)P(w)
$$

L1与L2理论涉及到朗格朗日优化。

$$
w=argmax(P(w|X))=argmax(P(X|w).P(w))
$$

从损失函数的角度，加了一个针对w大小的惩罚。

### L1正则

- 假设w的先验满足拉普拉斯分布
- 在损失函数中引入了绝对值，绝对值的导数是不连续的。可采用坐标下降法优化，即每次更新一个w的分量

### L2正则

- 假设w的先验服从零均值正态分布
- 解析解变为

$$
w=(X^TX+\lambda I)^{-1}X^TY
$$

## 4. 贝叶斯线性回归与高斯过程回归

贝叶斯线性回归要预测出一个关于y的分布来。稍微拓展一下贝叶斯线性回归到高斯过程回归(Gaussian Process Regression)

贝叶斯线性回归最大的不同在于预测的不是关于w的点估计，而是w的分布，即w的期望和协方差。

## 5. 问答

- pros:

  - explainable method
  - interpretable results by its output coefficients
  - faster to train than other machine learning models

- cons:

  - assumes linearity between inputs and outputs
  - sensitive to outliers
  - can under fit with small, high-dimensional data

- 为什么需要对数值类型特征进行归一化

  - 使用梯度下降优化的模型，归一化容易更快通过梯度下降找到最优解，包括线性回归、逻辑回归、支持向量机、神经网络。

- 如何判断是否该用线性回归模型

  - EDA画图
  - 检查预测值与真实值的residual残差是否为高斯分布

- Lasso regression如何导致参数稀疏

  - 可用于特征筛选
  - 原因: 直观上可以从图像切点理解，数学上可以通过朗格朗日优化解释

- 异常点如何处理

  - 检测并去除
  - Huber Loss is a loss function that combines MSE and MAE. It is less sensitive to outliers than a squared error loss and has the advantage of not needing to remove outlier data from the dataset

- 共线性问题 Multi-collinearity
  - 自变量之间由于存在高度相关关系(correlation)而使模型的权重参数估计失真或难以估计准确的一种特性
  - 共线性不影响模型的预测而是影响对模型的解释
  - 可以尝试l2正则；可以使用PCA，将特征转为独立的变量
  - feature selection, ridge regression, PCA

## 6. 代码

梯度下降

- 对多元函数的参数求∂偏导数，把求得的各个参数的偏导数以向量的形式写出来，就是梯度
- 从几何意义上讲，梯度就是函数变化增加最快的方向。减梯度就是朝着减小的方向

```python
def update_weights(w, b, x, y, learning_rate):
    w_grad = 0
    b_grad = 0
    N = len(x)
    for i in range(N):
        w_grad += -2 * x[i] * (y[i] - (w * x[i] + b))
        b_grad += -2 * (y[i] - (w * x[i] + b))

    w -= (w_grad / float(N)) * learning_rate
    b -= (b_grad / float(N)) * learning_rate
    return w, b
```

牛顿法

- 在现有极小点估计值的附近对 f(x) 做二阶泰勒展开，进而找到极小点的下一个估计值

```python
# https://www.stat.cmu.edu/~cshalizi/350/lectures/26/lecture-26.pdf
```

线性回归

```python
# 2018.01.26 我学习机器学习的第一天写的代码，learning_rate too large cause the gradient explosion
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

class LinearRegression(object):
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, x, y):
        batch_size, feature_size = x.shape
        self.w = np.random.random((feature_size, 1))
        self.b = np.random.random(1)
        self._gradient_descent(x, y)

    def eval(self):
        pass

    def predict(self, x):
        y_hat = np.dot(x, self.w) + self.b
        return y_hat

    def plot(self, x, y, y_pred):
        plt.plot(x, y, 'r-')
        plt.plot(x, y_pred, 'b.')
        plt.show()

    def _gradient_descent(self, x, y, learning_rate=0.0001, epoch=100):
        for i in range(epoch):
            y_hat = self.predict(x)  # => batch_size * 1
            loss = np.dot((y_hat - y).T, (y_hat - y))
            w_gradient = np.dot(x.T, x).dot(self.w) - np.dot(x.T, (y - self.b))
            b_gradient = self.b - np.mean(y - np.dot(x, self.w))
            self.w = self.w - learning_rate * w_gradient
            self.b=self.b - learning_rate * b_gradient
            print('step:', i, 'Loss:', loss[0][0])

    def __str__(self):
        return 'weights\t:%s\n bias\t:%f\n' % (self.w, self.b)

def generate_data():
    '''generate the examples to implement linear regression by numpy'''
    x = np.linspace(1, 30, num=30).reshape(-1, 3)
    y = np.linspace(1, 10, num=10) + np.random.random(10)
    y = y.reshape(-1, 1)
    return x,y

if __name__ == "__main__":
    x, y = generate_data()
    lr = LinearRegression()
    lr.fit(x, y)
    y_hat = lr.predict(x)
    lr.plot(x, y, y_hat)
```

## 参考

- [wikipedia-linear regression](https://en.wikipedia.org/wiki/Linear_regression)
- [stat.cmu.edu/~cshalizi/mreg/15/lectures/04/lecture-04](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/04/lecture-04.pdf)
- [Courses/ml/ppt/04_LinearRegression](http://dengcai.zjulearning.org.cn/Courses/ml/ppt/04_LinearRegression.pdf)
- [为什么L1和L2正则化可防止过拟合 - 大龙的文章 - 知乎](https://zhuanlan.zhihu.com/p/85630046)
- [5.1 - Ridge Regression](https://online.stat.psu.edu/stat508/lesson/5/5.1)
- [Gradient Descent Algorithm — a deep dive](https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21)
- [https://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/3-2-OLS.html](https://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/3-2-OLS.html)
