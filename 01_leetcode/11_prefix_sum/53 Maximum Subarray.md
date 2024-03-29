# 53 Maximum Subarray
[https://leetcode.com/problems/maximum-subarray/](https://leetcode.com/problems/maximum-subarray/)


## solution

- 前缀和+贪心
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSum = float('-inf')
        currentSum = 0
        
        for num in nums:
            currentSum += num
            
            if currentSum > maxSum:
                maxSum = currentSum
            
            if currentSum < 0:
                currentSum = 0
        
        return maxSum
```
时间复杂度：O(n) <br>
空间复杂度：O(1)


- 动态规划
  - 以nums[i]为结尾 的最大连续子序列和为dp[i]
  - 空间复杂度可以优化为O(1)
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return
        
        if len(nums) == 1:
            return nums[0]
        
        dp = [0] * len(nums)
        dp[0] = nums[0]

        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1] + nums[i], nums[i])
        return max(dp)
```
时间复杂度：O(n) <br>
空间复杂度：O(n)


## follow up
[560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/description/)

- 前缀和
```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        res = 0
        prefix_sum = 0
        d = {0: 1}

        for num in nums:
            prefix_sum = prefix_sum + num

            if prefix_sum - k in d:
                res += d[prefix_sum - k]
            if prefix_sum not in d:
                d[prefix_sum] = 1
            else:
                d[prefix_sum] += 1
        return res
```
时间复杂度：O(n) <br>
空间复杂度：O(n)


[918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/description/)
- 直接变两倍得到的结果是不正确的，还需要更精细的操作.
- 解法
  - 普通段: 如53
  - 首尾两段相连: 相当于中间扣掉一段最小, 然后剩余的即最大
```python
# 特别慢
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        if max(nums) <= 0:
            return max(nums)
        
        cur_min = nums[0]
        cur_max = nums[0]
        res = nums[0]

        for i in range(1, len(nums)):            
            cur_max = max(cur_max + nums[i], nums[i])
            res = max(res, cur_max)            
            cur_min = min(cur_min + nums[i], nums[i])
            res = max(res, sum(nums)-cur_min)       
        return res
```

```python
# 原理相同: 两个dp空间换时间
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        if max(nums) <= 0:
            return max(nums)

        max_dp = [i for i in nums]
        min_dp = [i for i in nums]
        for i in range(1, len(nums)):
            if max_dp[i-1] > 0:
                max_dp[i] += max_dp[i-1]
            if min_dp[i-1] < 0:
                min_dp[i] += min_dp[i-1]
        return max(max(max_dp), sum(nums) - min(min_dp))
```


[152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/description/)
```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        dp_pos = [nums[0]]
        dp_neg = [nums[0]]
        for i in range(1, len(nums)):
            dp_pos.append(max(nums[i], dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i]))
            dp_neg.append(min(nums[i], dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i]))
        return max(dp_pos)
```
