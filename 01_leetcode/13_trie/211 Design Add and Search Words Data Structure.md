# 211 Design Add and Search Words Data Structure
[https://leetcode.com/problems/design-add-and-search-words-data-structure/](https://leetcode.com/problems/design-add-and-search-words-data-structure/)


## solution

```python
class TrieNode:
    def __init__(self):
        self.data = {}
        self.is_word = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        for char in word:
            cur = cur.data.setdefault(char, TrieNode())
        cur.is_word = True        

    def search(self, word: str) -> bool:
        def dfs(node, index):
            if index == len(word):
                return node.is_word
            if word[index] == '.':
                for child in node.data.values():
                    if dfs(child, index + 1):
                        return True
            if word[index] in node.data:
                return dfs(node.data[word[index]], index+1)
            return False
        
        return dfs(self.root, 0)
```
时间复杂度：O() <br>
空间复杂度：O()
