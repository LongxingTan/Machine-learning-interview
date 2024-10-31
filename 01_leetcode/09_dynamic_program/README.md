# 动态规划

- 问题
  - 解决有**重复子问题**的**最优化**问题
  - 通过保存子问题的解，避免重复计算
  - 思路：数学归纳法，base case
    - 确定dp状态的含义
    - 递推方程与初始化。通过状态枚举帮助完成递推方程
    - 循环方式

- 类型
  - top-down (记忆化dfs)
  - bottom-up (循环)

- 类型
  - 记忆化搜索（DFS + Memoization Search）
  - for循环方式的动态规划，非Memoization Search方式。DP可以在多项式时间复杂度内解决DFS需要指数级别的问题。常见的题目包括找最大最小，找可行性，找总方案数等，一般结果是一个Integer或者Boolean
  - 结合了BFS和动态规划，或者DFS和动态规划
  - 结果: 有时候DP不是最后的结果，还需要用min/max/sum(dp)

- 常见
  - 背包问题(Knapsack problem)
    - 递推公式中根据本次加入的数据(重量)决定了向上查找的状态
    - 01背包
    - 完全背包 Unbounded Knapsack
  - 打劫问题
  - 字符串子序列
    - 编辑距离
    - 最长公共字串
    - 回文串
  - 股票问题


## 背包
01背包二维
```python
# dp[i][j]：从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少
# dp[i][j] = max(dp[i−1][j], dp[i−1][j−w[i]]+v[i]) // j >= w[i]

```

01背包一维
```python
# dp[0,...,W] = 0
# for i = 1,...,N
#     for j = W,...,w[i] // 必须逆向枚举!!!
#         dp[j] = max(dp[j], dp[j−w[i]]+v[i])
```

恰好装满
- 没有恰好装满背包的限制，将dp全部初始化成0就可以
- 如果有恰好装满的限制，那只应该将dp[0,...,N][0]初始为0，其它dp值均初始化为-inf

求方案总数
- dp[i][j] = sum(dp[i−1][j], dp[i][j−w[i]]) // j >= w[i]

求最优方案
- 一般动态规划问题输出方案的方法：记录下每个状态的最优值是由哪一个策略推出来的，这样便可根据这条策略找到上一个状态，从上一个状态接着向前推即可


## 参考
- [educative-io-contents](https://github.com/asutosh97/educative-io-contents/blob/master/Grokking%20Dynamic%20Programming%20Patterns%20for%20Coding%20Interviews.md)
- [Grokking-the-Coding-Interview-Patterns](https://github.com/cl2333/Grokking-the-Coding-Interview-Patterns-for-Coding-Questions)
- [动态规划之背包问题系列](https://tangshusen.me/2019/11/24/knapsack-problem/)
