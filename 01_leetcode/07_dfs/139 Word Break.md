# 139 Word Break
[https://leetcode.com/problems/word-break/](https://leetcode.com/problems/word-break/)


## solution

- dfs with memorization

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        memo = {}
        wordSet = set(wordDict)
        return self.dfs(s, wordSet, memo)
    
    def dfs(self, s, wordSet, memo):
        if s in memo:
            return memo[s]
        if s in wordSet:
            return True
        for i in range(1, len(s)):
            prefix = s[:i]
            if prefix in wordSet and self.dfs(s[i:], wordSet, memo):
                memo[s] = True
                return True
        memo[s] = False
        return False
```
时间复杂度：O() <br>
空间复杂度：O()


- 动态规划

```python
# dp[i]: s[i-1]是否可被word切割
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)        
        dp = [False] * (n + 1)
        dp[0] = True        

        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        return dp[-1]
```

```python
# - 完全背包
# - 1维dp: 容量为j的背包，所背的物品价值可以最大为dp[j]
# - 先物品还是先背包：因为不完全背包1维需要从后往前，
# - 背后从前到后还是从后到前
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        l = len(s)
        dp = [True] + [False] * l

        for i in range(1, l+1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
        return dp[-1]
```
时间复杂度：O() <br>
空间复杂度：O()


## follow up

[140. Word Break II](https://leetcode.com/problems/word-break-ii/)

- DFS枚举每次切词的位置，使用Memorization优化(回溯+动态规划思想)，每个字符串只用求一遍

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        memory = {}
        return self.dfs(s, wordDict, memory)
    
    def dfs(self, s, wordDict, memory):
        if s in memory:
            return memory[s]
        
        if not s:
            return []
        
        res = []
        for word in wordDict:
            if not s.startswith(word):
                continue
            if len(word) == len(s):
                res.append(word)
            else:
                rest_res = self.dfs(s[len(word):], wordDict, memory)
                for item in rest_res:
                    item = word + ' ' + item
                    res.append(item)
        memory[s] = res
        return res
```
时间复杂度：O() <br>
空间复杂度：O()


- Trie
```python
# https://algo.monster/liteproblems/140
```


[472 Concatenated Words](./472%20Concatenated%20Words.md)
