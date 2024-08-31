# 285 Inorder Successor in BST
[https://leetcode.com/problems/inorder-successor-in-bst/](https://leetcode.com/problems/inorder-successor-in-bst/)


## solution

```python
# BST下一个, 也就是找到第一个比p大的节点, 即最后一个访问的左节点
# 如果存在右节点，则右子树最左边的节点就是其下一个
# 如果不存在右节点，

class Solution(object):
    def inorderSuccessor(self, root, p):
        if not root:
            return
        
        successor = None
        while root:
            if root.val > p.val:
                successor = root
                root = root.left
            else:
                root = root.right
        return successor
```
时间复杂度：O(n) <br>
空间复杂度：O(h)


```python
# 如果根节点小于或等于要查找的节点, 直接进入右子树递归;如果根节点大于要查找的节点, 则暂存左子树递归查找的结果, 如果是 null, 则直接返回当前根节点; 反之返回左子树递归查找的结果.

class Solution:
    def inorderSuccessor(self, root, p):
        if root is None:
            return None
        
        if root.val <= p.val:
            return self.inorderSuccessor(root.right, p)
        
        left = self.inorderSuccessor(root.left, p)
        if left is None:
            return root
        else:
            return left
```


## follow up

[*510. Inorder Successor in BST II](https://leetcode.com/problems/inorder-successor-in-bst-ii/description/)
```python

```

[173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/description/)
```python
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left        

    def next(self) -> int:
        node = self.stack.pop()

        right = node.right
        while right:
            self.stack.append(right)
            right = right.left
        
        return node.val        

    def hasNext(self) -> bool:
        return len(self.stack) > 0
```
