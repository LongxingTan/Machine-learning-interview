# 39 Combination Sum
[https://leetcode.com/problems/combination-sum/](https://leetcode.com/problems/combination-sum/)


## solution

(可以换一种问法，类似k-sum，给定array和target，返回全部可以总和到target的group)

- 有放回+控制重复。注意理解这里的需求，有放回是当下的元素可以无限取，但不能从过去元素取，否则造成结果重复。（元素可以重复，结果不能重复）
- 回溯时，每次的开始start仍然由i进行，小于0时进行剪枝（因此需要排序）
- 寻找某个target的dfs，都是target-item来递归。回溯是否需要再加回去？

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []
        candidates.sort()
        self.dfs(path, res, candidates, target, start=0)
        return res

    def dfs(self, path, res, candidates, target, start):
        if target < 0:  # 一定要有这步？
            return

        if target == 0:
            res.append(path[:])
            return

        for i in range(start, len(candidates)):
            path.append(candidates[i])
            self.dfs(path, res, candidates, target - candidates[i], i)  # 有放回i, 无放回i+1
            path.pop()
```
时间复杂度：O(candidates^target) <br>
空间复杂度：O(target)


## follow up

[40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)
- 无放回，控制开始的index每次i+1，而不是i
- 控制重复，采用排序加剪枝。for循环横向遍历，对同一层使用过的元素跳过

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []
        start = 0
        candidates.sort(reverse=True)
        self.dfs(candidates, target, path, res, start)
        return res

    def dfs(self, candidates, target, path, res, start):
        if target < 0:
            return
        if target == 0:
            res.append(path[:])
            return

        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i-1]:  # 类似3sum中的控制重复
                continue
            path.append(candidates[i])
            self.dfs(candidates, target-candidates[i], path, res, i+1)
            path.pop()
```
时间复杂度：O(n⋅2^n) <br>
空间复杂度：O(n+n⋅2^n)


[216. Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)
- 不重复，开始从i+1
- 规定了选取的个数

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        path = []
        res = []
        self.dfs(path, res, k, n, start=1)
        return res

    def dfs(self, path, res, k, residual, start):
        if len(path) == k and residual == 0:
            res.append(path.copy())
            return

        for i in range(start, 10):
            path.append(i)
            residual -= i
            self.dfs(path, res, k, residual, i+1)
            path.pop(-1)
            residual += i
```
时间复杂度：O(n * 2^n) <br>
空间复杂度：O(n)


```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        path = []
        res = []
        start = 1
        return self.dfs(path, res, k, n, start)

    def dfs(self, path, res, k, n, start):
        if len(path) == k and sum(path) == n:
            res.append(path[:])

        for i in range(start, 10):
            path.append(i)
            self.dfs(path, res, k, n, i+1)
            path.pop()
        return res
```
时间复杂度：O() <br>
空间复杂度：O()


[377. Combination Sum IV](../09_dynamic_program/377%20Combination%20Sum%20IV.md)



[494. Target Sum](https://leetcode.com/problems/target-sum/)
- 动态规划/背包
```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        if abs(target) > total_sum:
            return 0
        if (target + total_sum) % 2 == 1:
            return 0
        
        target_sum = (total_sum + target) // 2
        dp = [0] * (target_sum + 1)
        dp[0] = 1

        for i in nums:
            for j in range(target_sum, i-1, -1):
                dp[j] += dp[j-i]
        return dp[target_sum]
```
