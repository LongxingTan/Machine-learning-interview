# 314 Binary Tree Vertical Order Traversal
[https://leetcode.com/problems/binary-tree-vertical-order-traversal/](https://leetcode.com/problems/binary-tree-vertical-order-traversal/)


## solution

- bfs，同时记录当前节点的垂直位置

```python
import collections


class Solution(object):
    def verticalOrder(self, root):
        if not root:
            return []

        res = collections.defaultdict(list)  # 注意结果保存的数据结构
        queue = collections.deque([(root, 0)])  # 注意初始化时每个item是（root, distance）的tuple
        while queue:
            for _ in range(len(queue)):
                node, distance = queue.popleft()
                res[distance].append(node.val)
                if node.left:
                    queue.append((node.left, distance - 1))
                if node.right:
                    queue.append((node.right, distance + 1))
        return [values for _, values in sorted(res.items())]
```
时间复杂度：O(n) <br>
空间复杂度：O(n)


## follow up

[987. Vertical Order Traversal of a Binary Tree](../07_dfs/987%20Vertical%20Order%20Traversal%20of%20a%20Binary%20Tree.md)
