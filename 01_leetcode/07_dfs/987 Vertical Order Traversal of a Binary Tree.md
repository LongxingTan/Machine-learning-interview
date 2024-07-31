# 987 Vertical Order Traversal of a Binary Tree
[https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)


## solution

```python
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        mydict = collections.defaultdict(list)
        self.dfs(root, 0, 0, mydict)
        result = []
        for i in sorted(mydict.keys()):
            temp = []
            for j in sorted(mydict[i]):  # top to bottom顺序，所以也要记录level
                temp.append(j[1])
            result.append(temp)
        return result
    
    def dfs(self, root, distance, level, mydict):
        if not root:
            return
        mydict[distance].append((level, root.val))
        self.dfs(root.left, distance - 1, level + 1, mydict)
        self.dfs(root.right, distance + 1, level + 1, mydict)
```


- 慢
```python
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        queue = collections.deque()
        queue.append([root, 0, 0])
        res = collections.defaultdict(list)
       
        while queue:
            node, distance, level = queue.popleft()
            res[distance].append((level, node.val))
            if node.left:
                queue.append([node.left, distance-1, level+1])
            if node.right:
                queue.append([node.right, distance+1, level+1])
        
        output = []
        for i in sorted(res.keys()):
            temp = []
            for j in sorted(res[i]):
                temp.append(j[1])
            output.append(temp)
        return output
```
时间复杂度：O(nlog(n)) <br>
空间复杂度：O(n)
