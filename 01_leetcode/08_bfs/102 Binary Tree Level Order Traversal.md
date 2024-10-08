# 102 Binary Tree Level Order Traversal
[https://leetcode.com/problems/binary-tree-level-order-traversal/](https://leetcode.com/problems/binary-tree-level-order-traversal/)


## solution

- 如果需要区分bfs的每一层，需要每一层更新队列；如果不区分每层，采用队列的push、pop

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return

        queue = collections.deque([root])
        res = []
        while queue:
            levels = []  # 通过这里，每层新开一个array，保存结果。对应需要队列每次都pop
            for _ in range(len(queue)):  # 让每层间隔开的关键在于这个for，而且size一定要保持刚进入循环时的size
                node = queue.popleft()
                levels.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(levels)
        return res
```
时间复杂度：O(n) <br>
空间复杂度：O(n)


## follow-up
[429. N-ary Tree Level Order Traversal](https://leetcode.com/problems/n-ary-tree-level-order-traversal/)

```python
"""
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if root is None:
            return []

        level = [root]
        res = []

        while level:
            this_level = []
            this_res = []

            for node in level:
                this_res.append(node.val)

                for child in node.children:
                    this_level.append(child)

            level = this_level
            res.append(this_res)
        return res
```
时间复杂度：O() <br>
空间复杂度：O()

```python
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []

        levels = collections.deque([root])
        res = []
        while levels:
            level_res = []
            for i in range(len(levels)):  # 按层bfs这里会有一个层的
                node = levels.popleft()
                level_res.append(node.val)
                for child in node.children:  # 层中每一个可选下一个
                    if child:
                        levels.append(child)
            res.append(level_res)
        return res
```

[107. Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/description/)

```python
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        queue = [root]
        res = []

        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            res.append(level)

        return res[::-1]
```
