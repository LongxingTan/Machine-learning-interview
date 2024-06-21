# 543 Diameter of Binary Tree
[https://leetcode.com/problems/diameter-of-binary-tree/](https://leetcode.com/problems/diameter-of-binary-tree/)


## solution

```python
class Solution:
    def __init__(self):
        self.res = 0

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.dfs(root)
        return self.res # 注意最终要输出过程中最大的需要一个额外global变量

    def dfs(self, root):  # 注意因为返回的因素，需要额外定义一个dfs
        if not root:
            return 0

        l = self.dfs(root.left)
        r = self.dfs(root.right)
        self.res = max(l + r, self.res)
        return max(l, r) + 1
```
时间复杂度：O(n) <br>
空间复杂度：O(h)


## follow up

[*1522. Diameter of N-Ary Tree](https://leetcode.com/problems/diameter-of-n-ary-tree/description/)
```python
# 可以使用优化, heapq.nlargest(2, [dfs(n) for n in node.neighbors])
class Solution:
    """
    @param root: the root of the tree
    @return: Maximum diameter of the N-ary Tree
    """
    def __init__(self):
        self.res = 0

    def diameter(self, root: UndirectedGraphNode) -> int:
        self.dfs(root)
        return self.res

    def dfs(self, root):
        if not root:
            return 0

        child_height = []
        for neighbor in root.neighbors:
            child_height.append(self.dfs(neighbor))

        child_height.sort(reverse=True)

        if len(child_height) == 1:
            self.res = max(self.res, child_height[0])
        elif len(child_height) > 1:
            if child_height[0] + child_height[1] + 1 > self.res:
                self.res = child_height[0] + child_height[1]
        else:
            child_height = [0]

        return max(child_height) + 1
```

[*1245. Tree Diameter](https://leetcode.com/problems/tree-diameter/description/)
- 经典做法: 两次dfs，从树中任意节点dfs走到距离最远的节点，然后从这个最远节点再dfs走到最远节点，两个节点距离就是树的直径
```python

```

[687. Longest Univalue Path](https://leetcode.com/problems/longest-univalue-path/description/)
```python

```
