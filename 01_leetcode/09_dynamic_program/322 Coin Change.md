# 322 Coin Change
[https://leetcode.com/problems/coin-change/](https://leetcode.com/problems/coin-change/)


## solution

- 完全背包
  - dp[j]：凑足总额为j所需钱币的最少个数为dp[j]

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:        
        dp = [float('inf') for _ in range(amount+1)]  # 恰好装满类的背包问题初始化
        dp[0] = 0

        for i in coins:  # 先物品
            for j in range(i, amount+1):  # 再背包(或者不能说背包，而是背包的约束), 背包大于物品
                if dp[j-i] != float('inf'):
                    dp[j] = min(dp[j], dp[j-i]+1)

        if dp[amount] == float('inf'):
            return -1
        return dp[amount]
```
时间复杂度：O(∣coins∣∣amount∣) <br>
空间复杂度：O(∣amount∣)


- bfs
```python

```
时间复杂度：O() <br>
空间复杂度：O()


- dfs
```python

```
时间复杂度：O() <br>
空间复杂度：O()


## follow up

[518. Coin Change II](https://leetcode.com/problems/coin-change-ii/)
