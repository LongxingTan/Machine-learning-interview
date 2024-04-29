# 329 Longest Increasing Path in a Matrix
[https://leetcode.com/problems/longest-increasing-path-in-a-matrix/](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)


## solution

- dfs + dp
  - top-down: 周围比自己小的加1, 记忆化搜索
  - bottem-up: 先排序，再搜索

```python
# https://blog.csdn.net/u013325815/article/details/105806262
class Solution:
    def __init__(self):
        self.dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0
        
        m = len(matrix)
        n = len(matrix[0])

        dp = [[0] * n for _ in range(m)]
        res = 0
        for i in range(m):
            for j in range(n):
                res = max(res, self.dfs(matrix, i, j, dp))
        return res
    
    def dfs(self, matrix, i, j, dp):
        if dp[i][j]:
            return dp[i][j]
        
        dp[i][j] = 1
        for x, y in self.dirs:
            new_i = i + x
            new_j = j + y
            if 0 <= new_i < len(matrix) and 0 <= new_j < len(matrix[0]) and matrix[new_i][new_j] > matrix[i][j]:
                dp[i][j] = max(dp[i][j], self.dfs(matrix, new_i, new_j, dp) + 1)
        return dp[i][j]
```
时间复杂度：O(mn) <br>
空间复杂度：O(mn)


## follow up

[691. Stickers to Spell Word](https://leetcode.com/problems/stickers-to-spell-word/description/)

- 记忆化DFS
```python

```

- 状态压缩BFS
```python

```
