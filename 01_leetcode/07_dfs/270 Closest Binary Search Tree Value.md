# 270 Closest Binary Search Tree Value
[https://leetcode.com/problems/closest-binary-search-tree-value/](https://leetcode.com/problems/closest-binary-search-tree-value/)


## solution

```python
class Solution(object):
    def closestValue(self, root, target):
        res = None
        cur = root
        
        while cur:
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
