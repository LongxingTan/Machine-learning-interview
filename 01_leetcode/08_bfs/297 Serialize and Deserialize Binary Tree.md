# 297 Serialize and Deserialize Binary Tree
[https://leetcode.com/problems/serialize-and-deserialize-binary-tree/](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)


## solution

- bfs: decode函数就类似构造树的题目

```python
class Codec:
    def serialize(self, root):
        if not root:  # 否则会输出一个#
            return ''   
            
        data = []
        queue = collections.deque([root])    
        while queue:
            node = queue.popleft()
            if node:
                data.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                data.append('#')

   
        return ','.join(data)

    def deserialize(self, data):
        if not data:
            return
        
        data = data.split(',')
        root = TreeNode(data[0])  
        queue = collections.deque([root])    
        # when you pop a node, its children will be at i and i+1
        for i in range(1, len(data), 2):
            node = queue.popleft()
            if data[i] != '#':
                node.left = TreeNode(data[i])
                queue.append(node.left)
            if data[i + 1] != '#':
                node.right = TreeNode(data[i + 1])
                queue.append(node.right)
        return root
```
时间复杂度：O(n) <br>
空间复杂度：O(n)

- dfs
```python

```


## follow up-序列化和压缩类题目

[*271. Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/)
```python

```


[*288. Unique Word Abbreviation](https://leetcode.com/problems/unique-word-abbreviation/description/)
```python

```

[320. Generalized Abbreviation](https://leetcode.com/problems/generalized-abbreviation/description/)
```python

```

[535. Encode and Decode TinyURL](https://leetcode.com/problems/encode-and-decode-tinyurl/description/)
```python

```

[394 Decode String](../07_dfs/394%20Decode%20String.md)
