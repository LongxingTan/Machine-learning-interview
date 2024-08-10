# 698 Partition to K Equal Sum Subsets
[https://leetcode.com/problems/partition-to-k-equal-sum-subsets/](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/)


## solution

```python

```
时间复杂度：O() <br>
空间复杂度：O()


## follow up

[473. Matchsticks to Square](https://leetcode.com/problems/matchsticks-to-square/description/)
- 回溯
```python

```


[416. 分割等和子集](https://leetcode.com/problems/partition-equal-subset-sum/)
- 动态规划/01背包
```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums) % 2 == 1:
            return False
        
        length = sum(nums) // 2  # 背包重量为 一半
        dp = [0] * (length + 1)

        for i in nums:  # 先物品
            for j in range(length, i-1, -1):  # 再背包，01背包倒序避免重复放入
                dp[j] = max(dp[j], dp[j-i]+i)
        return dp[-1] == length  # 如果恰好装满一半
```
时间复杂度：O(nk) <br>
空间复杂度：O(k)
