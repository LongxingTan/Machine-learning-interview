# 53 Maximum Subarray
[https://leetcode.com/problems/maximum-subarray/](https://leetcode.com/problems/maximum-subarray/)


## solution

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res = -float('inf')
        prefix_sum = 0

        for num in nums:
            prefix_sum = max(prefix_sum + num, num)  # 要么选这个数，要么不选
            res = max(res, prefix_sum)
        return res
```

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
        if len(nums) < 1:
            return 0

        prefix_sum = 0
        prefix_count_dict = collections.defaultdict(int)
        prefix_count_dict[0] = 1  # 注意初始化, 自身=k

        res = 0
        for num in nums:
            prefix_sum += num
            res += prefix_count_dict[prefix_sum - k]   # 注意和dict更新的顺序会影响结果

            prefix_count_dict[prefix_sum] += 1
        return res
```
时间复杂度：O(n) <br>
空间复杂度：O(n)


[523 Continuous Subarray Sum](./523%20Continuous%20Subarray%20Sum.md)


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
        dp_pos = [nums[0]]  # 最大
        dp_neg = [nums[0]]  # 最小，由于之前子序列可能是最小的负数，再突然遇到一个负数从而得到最大的子序列积
        for i in range(1, len(nums)):
            dp_pos.append(max(nums[i], dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i]))
            dp_neg.append(min(nums[i], dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i]))
        return max(dp_pos)
```


[1186. Maximum Subarray Sum with One Deletion](https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/description/)
- 动态规划
```python
# dp(i, d): 删除d个元素的arr[:i + 1]最大子序和

class Solution:
    def maximumSum(self, arr: List[int]) -> int:
        dp = [[0] * 2 for _ in range(len(arr))]

        dp[0][0] = arr[0]
        dp[0][1] = 0

        res = arr[0]
        for i in range(1, len(arr)):
            dp[i][0] = max(dp[i-1][0] + arr[i], arr[i])
            dp[i][1] = max(dp[i-1][1] + arr[i], dp[i-1][0])  # 删一个 (如果前面删一个则必须加上现在，如果删现在的则以前必须不能删)
            res = max(res, dp[i][0], dp[i][1])
        return res
```

- 前缀和+后缀和
```python

```

[713. Subarray Product Less Than K](https://leetcode.com/problems/subarray-product-less-than-k/)
```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        res = 0
        prefix_prod = 1
        l = 0 

        for r, num in enumerate(nums):
            prefix_prod *= num

            while l <= r and prefix_prod >= k:
                prefix_prod //= nums[l]
                l += 1
            
            res += r - l + 1  # 结果加上从l开始，以r结果的数量
        return res
```