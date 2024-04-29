# 285 Inorder Successor in BST
[https://leetcode.com/problems/inorder-successor-in-bst/](https://leetcode.com/problems/inorder-successor-in-bst/)


## solution

```python
# - 如果存在右节点，则右子树最左边的节点就是其下一个
# - 如果不存在右节点，
class Solution(object):
    def inorderSuccessor(self, root, p):
        if not root:
            return
```
时间复杂度：O(n) <br>
空间复杂度：O(h)


## follow up

[*510. Inorder Successor in BST II](https://leetcode.com/problems/inorder-successor-in-bst-ii/description/)

```python

```
