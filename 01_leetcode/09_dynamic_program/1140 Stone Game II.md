# 1140 Stone Game II
[https://leetcode.com/problems/stone-game-ii/](https://leetcode.com/problems/stone-game-ii/)


## solution

- 博弈论: minimax approach with memorization

```python
# 树中找最优路径: https://algo.monster/liteproblems/1140
class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        N = len(piles)
        self.dp = {}

        def recursiveStoneGame(start, M):            
            if start >= N:
                return 0
            
            # take all if possible
            if N - start <= 2 * M:
                return sum(piles[start:])
            
            # memoization
            if (start, M) in self.dp:
                return self.dp[(start, M)]

            my_score = 0
            total_score = sum(piles[start:])
            # the opponent can take [1, 2*M] stones
            for x in range(1, 2*M+1):
                # get opponent's score
                opponent_score = recursiveStoneGame(start+x, max(x, M))
                # maintains max my_score
                my_score = max(my_score, total_score - opponent_score)
                
            self.dp[(start, M)] = my_score
                
            return my_score
        return recursiveStoneGame(0, 1)
```
时间复杂度：O() <br>
空间复杂度：O()


- 区间DP
```python
# 常见转移方程: dp(i,j) = min{dp(i,k-1) + dp(k,j)} + w(i,j)   (i < k <= j)

class Solution(object):
    def stoneGameII(self, piles):
        n = len(piles)
        f = [[0] * (n + 1) for _ in range(n)]
        s = 0
        for i in range(n - 1, -1, -1):
            s += piles[i]
            for j in range(1, n + 1):
                if i + j * 2 >= n:
                    f[i][j] = s
                else:
                    for k in range(1, j * 2 + 1):
                        f[i][j] = max(f[i][j], s - f[i + k][max(j, k)])

        return f[0][1]
```
