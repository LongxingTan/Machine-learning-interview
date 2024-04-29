# 104 Maximum Depth of Binary Tree
[https://leetcode.com/problems/maximum-depth-of-binary-tree/](https://leetcode.com/problems/maximum-depth-of-binary-tree/)


## solution

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return max(l, r) + 1
```
时间复杂度：O(n) <br>
空间复杂度：O(1)


## follow-up

[111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

- BFS
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        level = [root]
        depth = 1
        while level:
            this_level = []
            for i in level:
                if i.left is None and i.right is None:
                    return depth
                if i.left:
                    this_level.append(i.left)
                if i.right:
                    this_level.append(i.right)
            level = this_level
            depth += 1

```
时间复杂度：O() <br>
空间复杂度：O()

- DFS
```python

```


[110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)

```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True

        l = self.height(root.left)
        r = self.height(root.right)
        return abs(l - r) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)

    def height(self, root):
        if root is None:
            return 0
        
        return max(self.height(root.left), self.height(root.right)) + 1
        
```
时间复杂度：O() <br>
空间复杂度：O()


[Balance a Binary Search Tree](https://www.geeksforgeeks.org/convert-normal-bst-balanced-bst/)
