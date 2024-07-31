# 212 Word Search II
[https://leetcode.com/problems/word-search-ii/](https://leetcode.com/problems/word-search-ii/)


## solution

- 回溯
```python

```
时间复杂度：O() <br>
空间复杂度：O()

- trie
```python
from collections import defaultdict

class Trie:
    def __init__(self):
        self.children = defaultdict(Trie)
        self.word = ""

    def insert(self, word):
        cur = self
        for c in word:
            cur = cur.children[c]
        cur.is_word = True
        cur.word = word

class Solution:
    def word_search_i_i(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = Trie()
        for word in words:
            trie.insert(word)

        def dfs(now, i1, j1):
            if board[i1][j1] not in now.children:
                return
            ch = board[i1][j1]
            now = now.children[ch]
            if now.word != "":
                ans.add(now.word)
            board[i1][j1] = "#"
            for i2, j2 in [(i1 + 1, j1), (i1 - 1, j1), (i1, j1 + 1), (i1, j1 - 1)]:
                if 0 <= i2 < m and 0 <= j2 < n:
                    dfs(now, i2, j2)
            board[i1][j1] = ch

        ans = set()
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                dfs(trie, i, j)
        return list(ans)
```


## follow up

[79. Word Search](../07_dfs/79.%20Word%20Search.md)
