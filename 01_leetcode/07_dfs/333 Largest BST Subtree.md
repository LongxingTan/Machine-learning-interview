# 333 Largest BST Subtree
[https://leetcode.com/problems/largest-bst-subtree/](https://leetcode.com/problems/largest-bst-subtree/)


## solution

```python
import math

class Solution:
    def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
        largest_bst_size = 0
        
        def dfs(root):
            if not root:
                return (math.inf, -math.inf, 0)
            
            left_min, left_max, left_size = dfs(root.left)
            right_min, right_max, right_size = dfs(root.right)
            
            nonlocal largest_bst_size
            if left_max < root.val < right_min:
                largest_bst_size = max(largest_bst_size, left_size + right_size + 1)
                return min(left_min, root.val), max(right_max, root.val), left_size + right_size + 1
            else:
                return (math.inf, -math.inf, 0)        

        dfs(root)
        return largest_bst_size
```
时间复杂度：O(n) <br>
空间复杂度：O(n)
