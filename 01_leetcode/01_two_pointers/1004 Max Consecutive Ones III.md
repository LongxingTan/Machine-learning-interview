# 1004 Max Consecutive Ones III
[https://leetcode.com/problems/max-consecutive-ones-iii/](https://leetcode.com/problems/max-consecutive-ones-iii/)


## solution

- 滑动窗口中的window里不超过k个0的最长子串

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        mydict = collections.defaultdict(int)

        l = 0
        res = 0
        for i, num in enumerate(nums):
            mydict[num] += 1
            while mydict[0] == k+1:
                mydict[nums[l]] -= 1
                l += 1
            res = max(res, i - l +1)
        return res
```
时间复杂度：O(n) <br>
空间复杂度：O(1)
