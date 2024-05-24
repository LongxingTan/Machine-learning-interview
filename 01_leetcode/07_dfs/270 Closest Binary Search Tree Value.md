# 270 Closest Binary Search Tree Value
[https://leetcode.com/problems/closest-binary-search-tree-value/](https://leetcode.com/problems/closest-binary-search-tree-value/)


## solution

```python
class Solution:
    def closestValue(self, root, target):
        upper = root
        lower = root

        while root:
            if root.val > target:
                upper = root  # 先记录再往后迭代
                root = root.left
            elif root.val < target:
                lower = root
                root = root.right
            else:
                return root.val

        if abs(upper.val - target) > abs(lower.val - target):
            return lower.val
        return upper.val
```


```python
# 一些BST题目甚至可以先中序存为list, 再从list中找到最近的
class Solution(object):
    def closestValue(self, root, target):
        res = None
        cur = root

        while cur:
            # 更直接的思路: 记录遍历过程中距离, 并记录更新距离最小的dian
            if res is None or abs(res - target) > abs(cur.val - target):
                res = cur.val

            if cur.val > target:
                cur = cur.left
            else:
                cur = cur.right
        return res
```
时间复杂度：O(h) <br>
空间复杂度：O(h)


## follow up

[272. Closest Binary Search Tree Value II](https://leetcode.com/problems/closest-binary-search-tree-value-ii/description/)

```python

```
