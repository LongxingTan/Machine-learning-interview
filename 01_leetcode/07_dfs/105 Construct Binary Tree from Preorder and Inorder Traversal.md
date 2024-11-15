# 105 Construct Binary Tree from Preorder and Inorder Traversal
[https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)


## solution

- 树的构建必须从root开始，再递归左右子树，也就是前序遍历形式
- 注意递归的形式和边界选取

```python
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


[889. Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/description/)
```python
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        node_count = len(preorder)
        if node_count == 0:
            return None
        
        root = TreeNode(preorder[0])
        if node_count == 1:
            return root        

        for i in range(node_count - 1):
            if postorder[i] == preorder[1]:
                root.left = self.constructFromPrePost(
                    preorder[1: 1 + i + 1], postorder[:i + 1]
                )
                root.right = self.constructFromPrePost(
                    preorder[1 + i + 1:], postorder[i + 1: -1]
                )
                return root
```


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

            while(q):  # 找到整套的括号作为左子树、右子树的分割点
                if s[k]=='(':
                    q.append('(')
                elif s[k]==')':
                    q.pop()
                k += 1
            node.left, node.right = dfs(s[i+1:k-1]), dfs(s[k+1:-1])
            return node
        return dfs(s)
```

[606. Construct String from Binary Tree](https://leetcode.com/problems/construct-string-from-binary-tree/description/)


[*1485 Clone Binary Tree With Random Pointer](./1485%20Clone%20Binary%20Tree%20With%20Random%20Pointer.md)


[133. Clone Graph](../08_bfs/133.%20Clone%20Graph.md)


[1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/)
```python
class Solution:
    def __init__(self):
        self.node_list = []
    
    def balanceBST(self, root: TreeNode) -> TreeNode:        
        self.dfs(root)
        new_root = self.build_tree(self.node_list)
        return new_root
    
    def dfs(self, root):
        if not root:
            return
        
        self.dfs(root.left)
        self.node_list.append(root)
        self.dfs(root.right)
    
    def build_tree(self, node_list):
        if not node_list:
            return
        
        mid = len(node_list) // 2
        root = node_list[mid]
        root.left = self.build_tree(node_list[:mid])
        root.right = self.build_tree(node_list[mid+1:])
        return root
```
