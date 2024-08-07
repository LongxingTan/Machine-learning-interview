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

- 状态压缩BFS(-> 最短路径): 关键在于状态的表征
```python
class Solution:
    def minStickers(self, stickers: List[str], target: str) -> int:
        queue = collections.deque([0])
        steps = 0
        n = len(target)
        visited = [False] * (2 ** n)  # 1 << n, n位的字符，从全是0到全是1共有2^n个状态
        visited[0] = True

        while queue:
            for _ in range(len(queue)):
                current_state = queue.popleft()

                # 全部位都是1代表已凑成target
                if current_state == 2 ** n - 1:
                    return steps

                for sticker in stickers:
                    next_state = current_state
                    sticker_count = collections.Counter(sticker)

                    for i, char in enumerate(target):
                        # If the character at position i is not yet added, and the sticker has the char.
                        if not (next_state & (1 << i)) and sticker_count[char]:
                            next_state |= 1 << i  # next_state += 1 << i
                            sticker_count[char] -= 1

                    # If the next state has not been visited, mark it as visited and add to the queue.
                    if not visited[next_state]:
                        visited[next_state] = True
                        queue.append(next_state)
            steps += 1
        return -1
```


```python
from typing import List
from collections import deque, Counter

class Solution:
    def minStickers(self, stickers: List[str], target: str) -> int:
        n = len(target)
        state_size = 1 << n  # 2 ** n, 如果target[i]已有，则为1，否则为0
        q = deque([0])
        dist = {0: 0}

        while q:
            now = q.popleft()
            for sticker in stickers:
                state = now
                cnt = Counter(sticker)
                for i, c in enumerate(target):
                    if now & (1 << i) == 0 and cnt[c] > 0:
                        cnt[c] -= 1
                        state += (1 << i)
                if state in dist:
                    continue
                q.append(state)
                dist[state] = dist[now] + 1
                if state == state_size - 1:
                    return dist[state]
        return -1
```
