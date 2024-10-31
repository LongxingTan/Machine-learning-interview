# 230 Kth Smallest element in a BST
[https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/)


## solution

```python
# 如果k作为参数传值，k -=1 并判断k == 0有问题，还是因为k本身是immutable
class Solution:
    def __init__(self):
        self.rank = 0
        self.res = 0
    
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:          
        self.dfs(root, k)
        return self.res
    
    def dfs(self, root, k):
        if not root:
            return         
        
        self.dfs(root.left, k)
        
        self.rank += 1
        if self.rank == k:
            self.res = root.val
            return   

        self.dfs(root.right, k)
```

- 注意递归过程中的返回值
```python
class Solution:
    def __init__(self):
        self.res = []

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.dfs(root, k)
        return self.res[-1]

    def dfs(self, root, k):
        if not root:
            return

        self.dfs(root.left, k)

        if len(self.res) == k:
            return
        self.res.append(root.val)
        
        self.dfs(root.right, k)
```
时间复杂度：O(n) <br>
空间复杂度：O()
