# 强化学习

- 马尔可夫过程：解决序列决策问题，理解状态之间的转移概率
- 目标：reward最大化。需要有更好的policy选择action, 通过让agent获得状态转移概率
- exploration和exploitation的平衡
- Q-learning，DQN，TRPO, PPO, soft actor critic

## 基本概念
- RL: 一个智能体（Agent）采取行动（Action）从而改变自己的状态（State）获得奖励（Reward）与环境（Environment）发生交互的循环过程
- rollout: 指的是从一个特定的状态按照某个策略进行一系列动作和状态转移
- Actor: 也就是Agent，Actor对于特定的任务，有自己的一个策略π，策略π用一个神经网络表示，其参数为θ. LLM语境下，即由prompt输出response


## 理论

贝尔曼最优方程

**类型**

- model-based
- model-free
  - Value Based: 状态+动作学习到一个value, value直接反应reward
  - Policy Based: 由状态学习到动作的分布，根据分布进行决策
  - Actor-Critic: Actor通过状态学习动作的分布，Critic根据动作和新的状态学习value评价
    - 为什么要有Critic? 强化学习的算法稳定性问题。加Critic，使用State-value function或者Action-value function稳定策略梯度的计算过程。更高级的算法是采用Advantage Function，也就是加上Baseline，增加梯度计算的稳定性。

## DQN

## 策略梯度 policy gradient

## PPO (Proximal Policy Optimization)

> - rlhf(Reward + PPO)是 online 学习方式，dpo 是 offline 学习方式
> - 策略梯度 -> actor-critic -> PPO

近端策略优化

- 两个网络，分别是Actor和Critic

## DPO

## GRPO
- 相比PPO: 省去Critic模型，仅依赖组内奖励统计量


## 问答

- on-policy和off-policy的区别是什么
- On-policy都有什么，SASA的公式和Q learning的公式什么差别，为什么没有max
- 解释一下DQN离散，DQNN（连续），有没有手写过
- DPO (off-policy) 为什么会在学习过程中training positive的概率和training negative的概率都同时下降？

  - 和采样的方式以及DPO loss组成相关. BT loss，maximize training set中positive和negative的gap

- RLHF & DPO
  - DPO did exclude some practical aspects of the RLHF method, e.g. pretraining gradients.
  - the theoretical arguments of DPO equivalence make some assumptions that don’t necessarily apply in practice
  - RLHF gives you a reusable reward model, which has practical uses and advantages. DPO doesn’t have useful intermediate product.
  - DPO works off preference, whereas desirable RL objectives could have many forms

```python
# RLHF 伪代码

for prompts in dataloader:

    # Stage 1: response生成
    batch = actor.generate_sequences(prompts)

    # Stage 2: 训练数据准备
    batch = critic.compute_values(batch)
    batch = reference.compute_log_prob(batch)
    batch = reward.compute_reward(batch)
    batch = compute_advantages(batch)

    # Stage 3: actor和critic训练
    critic_metrics = critic.update_critic(batch)
    actor_metrics = actor.update_actor(batch)
```

## reference

- [https://spinningup.openai.com/en/latest/](https://spinningup.openai.com/en/latest/)
- [理解Actor-Critic的关键是什么](https://zhuanlan.zhihu.com/p/110998399)
- [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/pdf/2307.04964.pdf)
- [常见强化学习方法总结 - marsggbo的文章 - 知乎](https://zhuanlan.zhihu.com/p/98962807)
- [通俗理解强化学习 - 绯红之刃的文章 - 知乎](https://zhuanlan.zhihu.com/p/664348944)
- [DPO 是如何简化 RLHF 的 - 朱小霖的文章 - 知乎](https://zhuanlan.zhihu.com/p/671780768)
- [强化学习应该怎么入门？ - Alex的回答 - 知乎](https://www.zhihu.com/question/622724204/answer/3220047569)
- [为什么ppo优于policy gradient? - 程序员眼罩的回答 - 知乎](https://www.zhihu.com/question/357056329/answer/3392670236)
- [https://datawhalechina.github.io/easy-rl/#/chapter1/chapter1](https://datawhalechina.github.io/easy-rl/#/chapter1/chapter1)
- https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-fall21/schedule.html
- [人人都能看懂的RL-PPO理论知识 - 猛猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/7461863937)
