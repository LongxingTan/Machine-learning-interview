# 198 House Robber
[https://leetcode.com/problems/house-robber/](https://leetcode.com/problems/house-robber/)


## solution

- 只用到两个状态的DP，可以进一步优化空间

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return max(nums)
        
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        return dp[-1]
```
时间复杂度：O(n) <br>
空间复杂度：O(n)


## follow up

[213 House Robber II](./213%20House%20Robber%20II.md)

[337. House Robber III](https://leetcode.com/problems/house-robber-iii/)
- 树形DP
  - 此外，还有三角形DP [120. Triangle](https://leetcode.com/problems/triangle/description/)
```python
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        dp = self.traversal(root)
        return max(dp)
    
    def traversal(self, root):
        if not root:
            return (0, 0)
        
        l = self.traversal(root.left)
        r = self.traversal(root.right)

        val0 = max(l[0], l[1]) + max(r[0], r[1])  # 不偷当前节点
        val1 = root.val + l[0] + r[0]  # 偷当前节点
        return [val0, val1]
```
时间复杂度：O() <br>
空间复杂度：O()
