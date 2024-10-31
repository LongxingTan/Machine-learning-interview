# 1268 Search Suggestions System
[https://leetcode.com/problems/search-suggestions-system/](https://leetcode.com/problems/search-suggestions-system/)


## solution

```python
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        res = []
        products.sort()

        for i in range(len(searchWord)):
            search = searchWord[:i+1]
            cur = []
            count = 0
            for prod in products:
                if search == prod[:i+1]:
                    cur.append(prod)
                    count += 1

                if count == 3:
                    break
            res.append(cur)
        return res
```
时间复杂度：O() <br>
空间复杂度：O()


- binary search
```python

```

- trie
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

class Solution:
    def suggestedProducts(self, products: list[str], searchWord: str) -> list[list[str]]:
        root = TrieNode()
        ans = []
        
        def insert(word):
            node = root
            for char in word:
                node = node.children.setdefault(char, TrieNode())
            node.word = word
        
        def search(node):
            res = []
            dfs(node, res)
            return res
        
        def dfs(node, res):
            if len(res) == 3:
                return
            if not node:
                return
            
            if node.word:
                res.append(node.word)
            
                for char in string.ascii_lowercase:
                    if char in node.children:
                        dfs(node.children[char], res)
            return
        
        for product in products:
            insert(product)
        
        node = root
        for char in searchWord:
            if not node or char not in node.children:
                node = None
                ans.append([])
                continue
            node = node.children[char]
            ans.append(search(node))
        return ans
```
