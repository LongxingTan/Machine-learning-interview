# 230 Kth Smallest element in a BST
[https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/)


## solution

- 注意递归过程中的返回值

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

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
时间复杂度：O() <br>
空间复杂度：O()
