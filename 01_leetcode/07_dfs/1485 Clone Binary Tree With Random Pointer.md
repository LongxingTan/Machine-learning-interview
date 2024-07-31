# 1485 Clone Binary Tree With Random Pointer
[https://leetcode.com/problems/clone-binary-tree-with-random-pointer/](https://leetcode.com/problems/clone-binary-tree-with-random-pointer/)


## solution

```python
class Solution:
    def clone_random_tree(self, root: RandomTreeNode) -> RandomTreeNode:
        if not root:
            return None

        copy = RandomTreeNode(root.val, None)
        copy.left = self.clone_random_tree(root.left)
        copy.right = self.clone_random_tree(root.right)
        if root.random:
            copy.random = self.clone_random_tree(root.random)

        return copy
```
时间复杂度：O() <br>
空间复杂度：O()


- **错误**
```python
class Solution:
    """
    @param root: The root node of a binary tree.
    @return: The cloned tree.
    """
    def clone_random_tree(self, root: RandomTreeNode) -> RandomTreeNode:
        # --- write your code here ---
        mydict = {}

        def dfs(root):
            if not root:
                return
            
            mydict[root] = RandomTreeNode(val=root.val)
            dfs(root.left)
            dfs(root.right)
        
        def dfs2(root):
            if not root:
                return
            
            mydict[root].random = mydict.get(root.random)
            mydict[root].left = mydict.get(root.left)
            mydict[root].right = mydict.get(root.right)

            dfs2(root.left)
            dfs2(root.right)
        
        dfs(root)
        dfs2(root)
        return mydict[root]
```