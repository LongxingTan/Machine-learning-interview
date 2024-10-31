# 304 Range Sum Query 2D - Immutable
[https://leetcode.com/problems/range-sum-query-2d-immutable/](https://leetcode.com/problems/range-sum-query-2d-immutable/)


## solution

- 把前缀和拓展到二维，即积分图(image integral)

```python
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])

        self.dp = [[0] * (n + 1) for _ in range(m + 1)]  # 考虑到后续计算，这里必须+1. 顺便简化了dp初始化      

        for i in range(m):
            for j in range(n):
                self.dp[i+1][j+1] = self.dp[i][j+1] + self.dp[i+1][j] - self.dp[i][j] + matrix[i][j]        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.dp[row2+1][col2+1] - self.dp[row1][col2+1] - self.dp[row2+1][col1] + self.dp[row1][col1]
```
时间复杂度：O(mn) <br>
空间复杂度：O(mn)
