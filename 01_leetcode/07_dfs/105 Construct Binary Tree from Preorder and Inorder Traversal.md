# 105 Construct Binary Tree from Preorder and Inorder Traversal
[https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)


## solution
- 树的构建必须从root开始，再递归左右子树，也就是前序遍历形式
- 注意递归的形式和边界选取

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return None
        
        root_val = preorder[0]
        root = TreeNode(root_val)

        split = inorder.index(root_val)

        inorder_left = inorder[:split]
        inorder_right = inorder[split+1:]

        preorder_left = preorder[1: len(inorder_left)+1]
        preorder_right = preorder[len(inorder_left)+1:]

        root.left = self.buildTree(preorder_left, inorder_left)
        root.right = self.buildTree(preorder_right, inorder_right)
        return root
```
时间复杂度：O(n) <br>
空间复杂度：O(n)


## follow up
[106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder or not postorder:  # 二者输入的长度必须相等
            return  # 等价于返回None

        root_val = postorder[-1]
        root = TreeNode(root_val)  # 构建树必须先构建root, 才有left和right

        split = inorder.index(root_val)  # 树的值必须是unique作为题目条件
        inorder_left = inorder[:split]
        inorder_right = inorder[split+1:]

        postorder_left = postorder[:len(inorder_left)]
        postorder_right = postorder[len(inorder_left): len(postorder) - 1]
        root.left = self.buildTree(inorder_left, postorder_left)
        root.right = self.buildTree(inorder_right, postorder_right)
        return root  # 整棵树的根节点最后出栈
```
时间复杂度：O(n) <br>
空间复杂度：O(n)


[1485 Clone Binary Tree With Random Pointer](./1485%20Clone%20Binary%20Tree%20With%20Random%20Pointer.md)

[133. Clone Graph](../08_bfs/133.%20Clone%20Graph.md)
