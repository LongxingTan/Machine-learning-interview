# 208 Implement Trie
[https://leetcode.com/problems/implement-trie-prefix-tree/](https://leetcode.com/problems/implement-trie-prefix-tree/)


## solution

- 根节点不包含字符，除根节点外每一个节点都只包含一个字符
- 从根节点到某一节点，路径上经过的字符连接起来，为该节点对应的字符串
- 每个节点的所有子节点包含的字符都不相同

```python
class Trie:
    def __init__(self):
        self.root = {}  # 多重的字典实现多叉树, 字典的value是另一个字典, 这个字典的多个key是下一层
        
    def insert(self, word: str) -> None:
        cur = self.root

        for letter in word:
            if letter not in cur:
                cur[letter] = {}
            cur = cur[letter]  # 关键在于理解这里
        cur['*'] = ''  # 结束标志

    def search(self, word: str) -> bool:
        cur = self.root
        for letter in word:
            if letter not in cur:
                return False
            cur = cur[letter]
        return '*' in cur
        
    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for letter in prefix:
            if letter not in cur:
                return False
            cur = cur[letter]
        return True
```
时间复杂度：O(l) <br>
空间复杂度：O()


```python
class TrieNode(object):
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_word = False


class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        """
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())            
        node.is_word = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for char in word:
            if char not in node.children:           
                return False
            node = node.children[char]
        return node.is_word

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for char in prefix:
            node = node.children.get(char)
            if not node:
                return False
        return True

    def get_start(self, prefix):
        """ 
        Returns words started with prefix
        """
        def get_key(pre, pre_node):
            word_list = []
            if pre_node.is_word:
                word_list.append(pre)
            for x in pre_node.children.keys():
                word_list.extend(get_key(pre + str(x), pre_node.children.get(x)))
            return word_list

        words = []
        if not self.startsWith(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for char in prefix:
            node = node.children.get(char)
        return get_key(prefix, node)
```
