# 472 Concatenated Words
[https://leetcode.com/problems/concatenated-words/](https://leetcode.com/problems/concatenated-words/)


## solution

- dfs [超时]
```python
# https://zhuanlan.zhihu.com/p/134658803

class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words.sort(key=len)
        prev = set()
        res = []
        for word in words:
            if self.dfs(prev, word):
                res.append(word)
            prev.add(word)
        return res

    def dfs(self, prev, word):
        if not prev:
            return False
        if not word:
            return True
        for i in range(1, len(word)+1):
            if word[:i] in prev and self.dfs(prev, word[i:]):
                return True
        return False
```
时间复杂度：O() <br>
空间复杂度：O()

- 动态规划
```python

```

- trie


## follow up

- 返回所有结果
