# 78 Subsets
[https://leetcode.com/problems/subsets/](https://leetcode.com/problems/subsets/)


## solution

- 空集是在哪里进入结果的？在第一次刚调用dfs之后就有了

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        path = []
        res =[]
        self.dfs(nums, path, res, start=0)
        return res

    def dfs(self, nums, path, res, start):
        res.append(path[:])
        if start == len(nums):
            return

        for i in range(start, len(nums)):
            path.append(nums[i])
            self.dfs(nums, path, res, i+1)
            path.pop()
```
时间复杂度：O(2^n) <br>
空间复杂度：O(n⋅2^n)


## follow up

[90. Subsets II](https://leetcode.com/problems/subsets-ii/)

- 重复数字需要去重
  - 排序，符合的ignore。可以看出来是对n叉树中的一行进行去重，去重方式与3sum类似

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        path = []
        res = []
        nums.sort()
        self.dfs(nums, path, res, start=0)
        return res

    def dfs(self, nums, path, res, start):
        res.append(path[:])

        if start >= len(nums):
            return

        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:  # 注意这里i > start, 而不是 i > 0
                continue

            path.append(nums[i])
            self.dfs(nums, path, res, i+1)  # 注意是i+1，不是start+1
            path.pop()
```
时间复杂度：O(n⋅2^n) <br>
空间复杂度：O(n⋅2^n)


[Count of subsets having sum of min and max element less than K](https://www.geeksforgeeks.org/count-of-subsets-having-sum-of-min-and-max-element-less-than-k/)
