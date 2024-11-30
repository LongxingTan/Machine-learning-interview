# ML数学基础

> 部分机器学习面试会直接考察统计与数学知识。即使不直接考察，在ML环节用数学佐证自己的观点也是很有裨益的。


## 1. 概率统计

- 中心极限定理
  - 中心极限定理指的是给定一个任意分布的总体。每次从这些总体中随机抽取 n 个抽样，一共抽 m 次。 然后把这 m 组抽样分别求出平均值。 这些平均值的分布接近正态分布。

- [Hypothesis testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
  - 通过样本来推测总体是否具备某种性质
  - 和最大似然类似？做出某个假设之后，依据其分布计算出，给出在这个分布下观察到这个现象的概率

- [z检验](https://en.wikipedia.org/wiki/Z-test)
  - 均值对比的假设检验方法主要有Z检验和T检验，Z检验面向总体数据和大样本数据，而T检验适用于小规模抽样样本

- [t检验/t-test](https://en.wikipedia.org/wiki/Student%27s_t-test)
  - t检验比z检验的普适性更强，z检验要求知道总体标准差，但实际研究中无法获知总体标准差，一般都会用t检验。且当样本量足够大的时候，数据接近正态分布，t检验几乎成为了z检验，z检验应该说t检验的一个特例

- [F-test](https://en.wikipedia.org/wiki/F-test#:~:text=An%20F-test%20is%20any,which%20the%20data%20were%20sampled.)

- P-value
  - 在假设原假设H0正确时，出现当前证据或更强的证据的概率

- confidence interval

- correlation matrix

- VIF

- R2/ adjusted R2

- ANOVA

- 蒙特卡洛


## 2. 矩阵

特征值与特征向量

迹(trace)
- 主对角线上的元素之和
- 矩阵的迹与特征值之和有关
- 协方差矩阵的迹是样本方差的和


## 3. 微积分

机器学习中使用的微积分主要在于优化和反向传播。


## 4. 问答

- What is p-value? What is confidence interval? Explain them to a product manager or non-technical person.
- How do you understand the "Power" of a statistical test?
- If a distribution is right-skewed, what's the relationship between medium, mode, and mean?
- When do you use T-test instead of Z-test? List some differences between these two.
- Dice problem-1: How will you test if a coin is fair or not? How will you design the process(有时会要求编程实现)? what test would you use?
- Dice problem-2: How to simulate a fair coin with one unfair coin?
- 3 door questions.
- Bayes Questions:Tom takes a cancer test and the test is advertised as being 99% accurate: if you have cancer you will test positive 99% of the time, and if you don't have cancer, you will test negative 99% of the time. If 1% of all people have cancer and Tom tests positive, what is the prob that Tom has the disease? (非常经典的cancer screen的题，做会这一道，其他都没问题了)
- How do you calculate the sample size for an A/B testing?
  - 确定显著性水平 α 和统计功效 1−β，常见选择是0.05和0.8
- If after running an A/B testing you find the fact that the desired metric(i.e, Click Through Rate) is going up while another metric is decreasing(i.e., Clicks). How would you make a decision?
- Now assuming you have an A/B testing result reflecting your test result is kind of negative (i.e, p-value ~= 20%). How will you communicate with the product manager? If given the above 20% p-value, the product manager still decides to launch this new feature, how would you claim your suggestions and alerts?
- 给定visitors and conversations，怎么计算significance
- 什么是type I/II error
- 圆周上任取三个点，能组成锐角三角形的概率是多大？
- rejection sampling
- [假设现有一枚均匀硬币，现要投掷硬币，直到其两次出现正面，求投掷的期望次数](https://zhuanlan.zhihu.com/p/64262250)
- Frequentists vs. Bayesians
  - One is called the frequentist interpretation. In this view, probabilities represent long run frequencies of events. For example, the above statement means that, if we flip the coin many times, we expect it to land heads about half the time.
  - The other interpretation is called the Bayesian interpretation of probability. In this view, probability is used to quantify our uncertainty about something; hence it is fundamentally related to information rather than repeated trials. In the Bayesian view, the above statement means we believe the coin is equally likely to land heads or tails on the next toss
  - One big advantage of the Bayesian interpretation is that it can be used to model our uncertainty about events that do not have long term frequencies. For example, we might want to compute the probability that the polar ice cap will melt by 2020 CE. This event will happen zero or one times, but cannot happen repeatedly. Nevertheless, we ought to be able to quantify our uncertainty about this event. To give another machine learning oriented example, we might have observed a “blip” on our radar screen, and want to compute the probability distribution over the location of the corresponding target (be it a bird, plane, or missile). In all these cases, the idea of repeated trials does not make sense, but the Bayesian interpretation is valid and indeed quite natural. We shall therefore adopt the Bayesian interpretation in this book. Fortunately, the basic rules of probability theory are the same, no matter which interpretation is adopted.


## Reference
- [https://brilliant.org/](https://brilliant.org/)
- [Udacity AB test](https://www.udacity.com/course/ab-testing--ud257)
- [A/B测试(AB实验)的基础、原理、公式推导、Python实现和应用 - 刘启林的文章 - 知乎](https://zhuanlan.zhihu.com/p/346602966)
- [bittiger AB test](https://drive.google.com/file/d/15F2lpaW0wGU5WK9lR46ygUsBReYig_YE/view)
- [Lets Python do AB testing analysis](https://github.com/tlentali/leab)
- [https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra](https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra)
- [https://github.com/kojino/120-Data-Science-Interview-Questions](https://github.com/kojino/120-Data-Science-Interview-Questions)
- [概率论——大数定律与中心极限定理](https://zhuanlan.zhihu.com/p/259280292)
- [线性代数|机器学习 18.065 by Gilbert Strang](https://www.bilibili.com/video/BV1a7411M7wH/)
- [矩阵求导术](https://zhuanlan.zhihu.com/p/24709748)
- [如何提供一个可信的AB测试解决方案](https://tech.meituan.com/2023/08/24/ab-test-practice-in-meituan.html)
- [假设检验——这一篇文章就够了](https://mp.weixin.qq.com/s/Klj7B2CMO3MF_O-HBfnddw)
- [马上去念CS PhD了，想补点数学课，请问有什么建议？ - CKLSniper的回答 - 知乎](https://www.zhihu.com/question/631954972/answer/3303408469)
- [为什么分母从n变成n-1之后，就从【有偏估计】变成了【无偏估计】？ - 包遵信的回答 - 知乎](https://www.zhihu.com/question/38185998/answer/76525265)
- [ABTest知识总结 - Steven的文章 - 知乎](https://zhuanlan.zhihu.com/p/450660183)
- [冠军/挑战者试验（A/B Test） - 过一点画一条直线的文章 - 知乎](https://zhuanlan.zhihu.com/p/144924899)
- [如何通俗地理解协方差和相关系数？ - 马同学的文章 - 知乎](https://zhuanlan.zhihu.com/p/70644127)
- A practical guide to quantitative finance interviews
- [P-Value](https://zhuanlan.zhihu.com/p/23806765)
