# 523 Continuous Subarray Sum
[https://leetcode.com/problems/continuous-subarray-sum/](https://leetcode.com/problems/continuous-subarray-sum/)


## solution

- 暴力(超时)
  - 另外，prefix的精华在于两个不同位置的prefix相减
```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        if len(nums) < 2:
            return False

        prefix_sum = [nums[0]]
        for i in range(1, len(nums)):
            news = []
            for pre in prefix_sum:
                new = pre + nums[i]
                if new % k == 0:
                    return True
                news.append(new)
            news.append(nums[i])
            prefix_sum = news
        return False
```
时间复杂度：O() <br>
空间复杂度：O()


- 根据mode的数学特点, 前缀+贪心
```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        if len(nums) < 2:
            return False

        prefix_sum = 0
        sum = {0: -1} # 遇到mode为0直接为True

        for i in range(len(nums)):
            prefix_sum += nums[i]
            remainder = prefix_sum % k
            # 如果两个数除k都余同一个a，那么二者之差可以整除k
            if remainder in sum and i - sum[remainder] > 1:
                return True
            if remainder not in sum:
                sum[remainder] = i
        return False
```


## follow up

[525. Contiguous Array](https://leetcode.com/problems/contiguous-array/description/)
- tricky的地方也在于，类似mode，通过hash记录了状态(而且是记录最早状态)，然后当前状态与hash中状态可以组成目标

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        prefix_sum = 0
        prefix_hash = {0: -1}  # 初始化解决了 f([0, 1]) = 2
        res = 0
        for i, num in enumerate(nums):
            if num == 1:
                prefix_sum += 1
            else:
                prefix_sum -= 1

            if prefix_sum in prefix_hash:
                res = max(res, i - prefix_hash[prefix_sum])
            else:
                prefix_hash[prefix_sum] = i
        return res
```
