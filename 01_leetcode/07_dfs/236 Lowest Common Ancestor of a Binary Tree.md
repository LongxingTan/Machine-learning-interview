# 236 Lowest Common Ancestor of a Binary Tree
[https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)


## solution

- 后续遍历，自底向上的回溯。
- 沿着树递归，注意如何最终返回最近公共祖先
- 理解递归终止单个的返回，与数多分支处理的返回

```python
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

- 返回两个目标node之间的路‍‌径
  - 分别求从LCA到两个node之间的路径，然后merge


## follow up

[tree node 可以直接call parent or child](https://www.geeksforgeeks.org/lowest-common-ancestor-in-a-binary-tree-using-parent-pointer/)

[*1644. Lowest Common Ancestor of a Binary Tree II](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/)
```python
# 转化为树直径的结构, init记录两个是否见过p与是否见过q的变量

```

[*1650. Lowest Common Ancestor of a Binary Tree III](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/)
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
# O(1) space, 类似160的思路
class Solution(object):
    def lowestCommonAncestor(self, p, q):
        a = p
        b = q

        while a != b:
            # If pointer_a has a parent, move to the parent; otherwise, go to the other node's initial position.
            a = a.parent if a else q
            b = b.parent if b else p
        return a
```
时间复杂度：O(h)
空间复杂度：O(1)


- 迭代
```python

```

[*1676. Lowest Common Ancestor of a Binary Tree IV](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iv/description/)
```python

```


[1123. Lowest Common Ancestor of Deepest Leaves](https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/description/)
```python
# 865. Smallest Subtree with all the Deepest Nodes
# 关键是转化为: 左右节点的性质来判断

class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        max_depth, root = self.dfs(root)
        return root

    def dfs(self, root):
        if not root:
            return 0, None

        left_depth, left_node = self.dfs(root.left)
        right_depth, right_node = self.dfs(root.right)

        if left_depth == right_depth:
            return left_depth + 1, root
        elif left_depth > right_depth:
            # 因为要返回最深, 如果左边最深，那么就是把左边返回的某parent节点往上传输
            return left_depth + 1, left_node
        else:
            return right_depth + 1, right_node
```
