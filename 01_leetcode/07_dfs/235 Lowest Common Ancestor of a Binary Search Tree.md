# 235 Lowest Common Ancestor of a Binary Search Tree
[https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)


## solution

- 思路是找到第一个位于p、q之间的节点。所以虽然是BST，但前序遍历

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)  # 这种遍历一半的树写法，判断然后返回。另一种完整遍历树，左结果l和右结果r再判断
        elif root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:  # 这里的else就包含两种情况，一种是p<=root<=q, 一种是q<=root<=p
            return root
```
时间复杂度：O(h)
空间复杂度：O(h)


- 迭代
```python

```
