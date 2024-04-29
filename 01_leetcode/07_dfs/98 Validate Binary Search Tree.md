# 98 Validate Binary Search Tree
[https://leetcode.com/problems/validate-binary-search-tree/](https://leetcode.com/problems/validate-binary-search-tree/)


## solution

- 注意二叉搜索树定义：根节点大于所有左子树，小于所有右子树。单个树递归的做法是无法满足定义的
- 递归也可以采用直观解法思路：中序展开为数组，判断数组是否排好序

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def __init__(self):
        self.pre = None

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        
        # 注意左右需要返回
        left = self.isValidBST(root.left)

        if self.pre is not None and self.pre >= root.val:
            return False
        self.pre = root.val

        right = self.isValidBST(root.right)
        return left and right
```
时间复杂度：O(n) <br>
空间复杂度：O(h)


- 层序遍历
```python

```

- 后序遍历
```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def bst(root, min_val=float('-inf'), max_val=float('inf')):
            if root == None:
                return True

            if not (min_val < root.val < max_val):
                return False

            return (bst(root.left, min_val, root.val) and
                    bst(root.right, root.val, max_val))

        return bst(root)
```


## follow up

[958. Check Completeness of a Binary Tree](https://leetcode.com/problems/check-completeness-of-a-binary-tree/description/)
- bfs: 一个是否最后一层的flag (节点个数), 一个是否应该stop的flag (是否有空的左或右节点)
```python
# https://zhuanlan.zhihu.com/p/360523724

```

- bfs: None也加入到queue中, 最后把None pop出去，检查是否还有额外node.
- bfs: 记录层序的上一个node, 如果现在node非空，上一个node为None,则False
```python
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        
        queue = collections.deque([root])    
        while queue[0] is not None:           
            node = queue.popleft()                
            # 显著不同是: 把None也加进去, 注意判断条件的改变
            queue.append(node.left)
            queue.append(node.right)
        
        for node in queue:
            if node:
                return False
        return True
```
