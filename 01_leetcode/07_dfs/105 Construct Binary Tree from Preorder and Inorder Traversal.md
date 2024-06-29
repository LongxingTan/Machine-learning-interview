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

[*536. Construct Binary Tree from String](https://leetcode.com/problems/construct-binary-tree-from-string/description/)
```python
# 树的构造: 就是先从root开始, 再构建左子树、右子树
# 同时参考 前中序构建二叉树, 394 decode string, 以及计算器题目

class Solution:
    def str2tree(self, s: str) -> TreeNode:
        def dfs(s):
            if not s:
                return None

            if '(' not in s:
                return TreeNode(int(s))

            i = s.index('(')
            node = TreeNode(int(s[:i]))  # 第一个左括号以左是根节点
            q = ['(']
            k = i + 1

            while(q):  # 找到完整的括号和左子树
                if s[k]=='(':
                    q.append('(')
                elif s[k]==')':
                    q.pop()
                k += 1
            node.left, node.right = dfs(s[i+1:k-1]), dfs(s[k+1:-1])
            return node
        return dfs(s)
```


[*1485 Clone Binary Tree With Random Pointer](./1485%20Clone%20Binary%20Tree%20With%20Random%20Pointer.md)


[133. Clone Graph](../08_bfs/133.%20Clone%20Graph.md)


[1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/)
```python
class Solution:
    def __init__(self):
        self.nodes = []

    def balanceBST(self, root: TreeNode) -> TreeNode:
        self.dfs(root)
        self.nodes = sorted(self.nodes, key=lambda x: x.val)
        return self.construct(self.nodes)

    def dfs(self, root):
        if not root:
            return

        self.nodes.append(root)
        if root.left:
            self.dfs(root.left)
        if root.right:
            self.dfs(root.right)

    def construct(self, nodes):
        if not nodes:
            return

        idx = len(nodes) // 2
        root = nodes[idx]
        root.left = self.construct(nodes[:idx])
        root.right = self.construct(nodes[idx+1:])
        return root
```
