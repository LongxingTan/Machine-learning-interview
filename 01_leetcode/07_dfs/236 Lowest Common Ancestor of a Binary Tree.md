# 236 Lowest Common Ancestor of a Binary Tree
[https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)


## solution

- 后续遍历，自底向上的回溯。
- 沿着树递归，注意如何最终返回最近公共祖先
- 理解递归终止单个的返回，与数多分支处理的返回

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None or root == p or root == q:
            return root

        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)

        if l and r:
            return root
        if l:
            return l
        if r:
            return r
```
时间复杂度：O(h)
空间复杂度：O(h)

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return
        
        if root == p:  # 根据自身节点的性质, 节点返回
            return p
        
        if root == q:
            return q
        
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)

        if l and r:  # 根据左右子树的结果, 节点返回
            return root
        if not l and r:
            return r
        if not r and l:
            return l
```

- 迭代
```python

```


## follow up
- [tree node 可以直接call parent or child](https://www.geeksforgeeks.org/lowest-common-ancestor-in-a-binary-tree-using-parent-pointer/)

- [1644. Lowest Common Ancestor of a Binary Tree II]()

- [*1650. Lowest Common Ancestor of a Binary Tree III](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/)

```python
# 160. Intersection of Two Linked Lists
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class Solution(object):
    def lowestCommonAncestor(self, p, q):      
        node_set = set()
        while p:
            node_set.add(p)
            p = p.parent
        
        while q not in node_set:
            q = q.parent
        
        return q
```

```python
# O1 space
class Solution(object):
    def lowestCommonAncestor(self, p, q):    
        a = p
        b = q
    
        while a != b:
          a = a.parent if a else q
          b = b.parent if b else p
    
        return a
```


- 迭代
```python

```

- [1676. Lowest Common Ancestor of a Binary Tree IV]()

- [1123. Lowest Common Ancestor of Deepest Leaves]()
