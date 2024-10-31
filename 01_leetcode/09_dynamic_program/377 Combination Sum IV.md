# 377 Combination Sum IV
[https://leetcode.com/problems/combination-sum-iv/](https://leetcode.com/problems/combination-sum-iv/)


## solution

- 完全背包|求组合方案个数
```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        # 排列: 先物品再背包
        # 组合: 先背包在物品 (本题)
        for i in range(1, target+1):
            for num in nums:
                if i - num >= 0:
                    dp[i] += dp[i - num]
        return dp[target]
```
时间复杂度：O(∣coins∣⋅target) <br>
空间复杂度：O(target)

- dfs
```python

```
