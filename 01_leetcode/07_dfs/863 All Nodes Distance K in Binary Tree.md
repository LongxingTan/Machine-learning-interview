# 863 All Nodes Distance K in Binary Tree
[https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)


## solution

```python
# 3个方向: 左子树找，右子数找，parent方向找. 3个方向parent还会向下因此需要记录是否visited
# 因此需要先建立一个 {node: parent}的map, 记录每个node 的parent.

class Solution:
    def __init__(self):
        self.res = []

    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        parent_map = {}

        def build_parent_map(root, parent):
            if not root:
                return

            parent_map[root] = parent
            build_parent_map(root.left, root)
            build_parent_map(root.right, root)

        build_parent_map(root, None)

        visited = set()
        self.dfs(target, k, parent_map, visited)
        return self.res

    def dfs(self, root, k, parent_map, visited):
        if not root:
            return

        if k == 0:
            self.res.append(root.val)
        visited.add(root)

        if root.left not in visited:
            self.dfs(root.left, k-1, parent_map, visited)
        if root.right not in visited:
            self.dfs(root.right, k-1, parent_map, visited)
        if parent_map[root] not in visited:
            self.dfs(parent_map[root], k-1, parent_map, visited)
```
时间复杂度：O(n) <br>
空间复杂度：O(n)
