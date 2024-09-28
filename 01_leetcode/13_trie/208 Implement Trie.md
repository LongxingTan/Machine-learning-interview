# 208 Implement Trie
[https://leetcode.com/problems/implement-trie-prefix-tree/](https://leetcode.com/problems/implement-trie-prefix-tree/)


## solution

- 根节点不包含字符，除根节点外每一个节点都只包含一个字符
- 从根节点到某一节点，路径上经过的字符连接起来，为该节点对应的字符串
- 每个节点的所有子节点包含的字符都不相同

```python
class Trie:
    def __init__(self):
        self.root = {}  # 多重字典实现多叉树, 字典的value是另一个字典, 这个字典的多个key是下一层

    def insert(self, word: str) -> None:
        cur = self.root  # cur类似在多重字典中的list指针

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
class Node:
    def __init__(self):
        self.children = collections.defaultdict(Node)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = Node()        

    def insert(self, word: str) -> None:
        cur = self.root
        for char in word:
            # cur = cur.children.setdefault(char, Node())  # set_default: 查找key, 存在则返回对应value；不存在，则在字典中添加key，并将值设置为指定值
            if char not in cur.children:
                cur.children[char] = Node()
            cur = cur.children[char]  # 没有新建，有则往下
        
        cur.is_word = True        

    def search(self, word: str) -> bool:
        cur = self.root
        for char in word:
            if char not in cur.children:
                return False
            cur = cur.children[char]
        return cur.is_word        

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for char in prefix:
            if char not in cur.children:
                return False
            cur = cur.children[char]
        return True
```
