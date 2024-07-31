# 314 Binary Tree Vertical Order Traversal
[https://leetcode.com/problems/binary-tree-vertical-order-traversal/](https://leetcode.com/problems/binary-tree-vertical-order-traversal/)


## solution

- bfs，同时记录当前节点的垂直位置

```python
import collections

class Solution(object):
    def verticalOrder(self, root):
        if not root:  # 注意边界
            return []

        # queue = collections.deque([(root, 0)])  # 注意初始化时每个item是（root, distance）的tuple
        queue = collections.deque()
        queue.append((root, 0))
        res_dict = collections.defaultdict(list)  # 注意结果保存的数据结构

        while queue:
            for _ in range(len(queue)):
                node, distance = queue.popleft()
                res_dict[distance].append(node.val)
                if node.left:
                    queue.append((node.left, distance - 1))
                if node.right:
                    queue.append((node.right, distance + 1))

        res = []
        for key in sorted(res_dict):  # 可以直接对字典的key进行排序
            res.append(res_dict[key])
        return res
```
时间复杂度：O(nlog(n)) <br>
空间复杂度：O(n)

- 时间复杂度O(n)优化


## follow up

[987. Vertical Order Traversal of a Binary Tree](../07_dfs/987%20Vertical%20Order%20Traversal%20of%20a%20Binary%20Tree.md)
