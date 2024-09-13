# 212 Word Search II
[https://leetcode.com/problems/word-search-ii/](https://leetcode.com/problems/word-search-ii/)


## solution

- trie
```python
# https://leetcode.com/problems/word-search-ii/solutions/59790/python-dfs-solution-directly-use-trie-implemented/

class Trie:
    def __init__(self):
        self.children = collections.defaultdict(Trie)
        self.word = ""

    def insert(self, word):  # 单词插入一个word
        cur = self  # 起点
        for c in word:
            cur = cur.children[c]
        cur.is_word = True
        cur.word = word

class Solution:
    def word_search_i_i(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = Trie()
        for word in words:
            trie.insert(word)

        def dfs(cur, i1, j1):
            if board[i1][j1] not in cur.children:
                return
            char = board[i1][j1]
            cur = cur.children[char]
            if cur.word != "":
                ans.add(cur.word)
            board[i1][j1] = "#"
            for i2, j2 in [(i1 + 1, j1), (i1 - 1, j1), (i1, j1 + 1), (i1, j1 - 1)]:
                if 0 <= i2 < m and 0 <= j2 < n:
                    dfs(cur, i2, j2)
            board[i1][j1] = char

        ans = set()
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                dfs(trie, i, j)
        return list(ans)
```

- 回溯
```python

```
时间复杂度：O() <br>
空间复杂度：O()


## follow up

[79. Word Search](../07_dfs/79.%20Word%20Search.md)
