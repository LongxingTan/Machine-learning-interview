# 强化学习

- 马尔可夫过程：解决序列决策问题，理解状态之间的转移概率
- 目标：reward最大化。需要有更好的policy选择action, 通过让agent获得状态转移概率
- exploration和exploitation的平衡
- Q-learning，DQN，TRPO, PPO, soft actor critic

## DQN


## 策略梯度 policy gradient


## PPO
rlhf(Reward + PPO)是 online 学习方式，dpo 是 offline 学习方式


## 问答
- 是on-policy还是off-policy的区别是什么
- On-policy都有什么，SASA的公式和Q learning的公式什么差别，为什么没有max
- 解释一下DQN离散，DQNN（连续），有没有手写过
- DPO (off-policy) 为什么会在学习过程中training positive的概率和training negative的概率都同时下降？
  - 和采样的方式以及DPO loss组成相关. BT loss，maximize training set中positive和negative的gap


## reference
- [理解Actor-Critic的关键是什么](https://zhuanlan.zhihu.com/p/110998399)
- [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/pdf/2307.04964.pdf)
- [常见强化学习方法总结 - marsggbo的文章 - 知乎](https://zhuanlan.zhihu.com/p/98962807)
- [通俗理解强化学习 - 绯红之刃的文章 - 知乎](https://zhuanlan.zhihu.com/p/664348944)
- [DPO 是如何简化 RLHF 的 - 朱小霖的文章 - 知乎](https://zhuanlan.zhihu.com/p/671780768)
- 