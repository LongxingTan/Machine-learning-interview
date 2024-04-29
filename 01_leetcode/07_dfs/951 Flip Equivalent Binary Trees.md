# 951 Flip Equivalent Binary Trees
[https://leetcode.com/problems/flip-equivalent-binary-trees/](https://leetcode.com/problems/flip-equivalent-binary-trees/)


## solution

```python
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        if root1 is None and root2 is None:
            return True
        if root1 is None or root2 is None:
            return False
        if root1.val != root2.val:
            return False
        return ((self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right))
                or (self.flipEquiv(root1.right, root2.left) and self.flipEquiv(root1.left, root2.right)))        
```
时间复杂度：O(n) <br>
空间复杂度：O(h)
