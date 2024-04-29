# 348 Design Tic-Tac-Toe
[https://leetcode.com/problems/design-tic-tac-toe/](https://leetcode.com/problems/design-tic-tac-toe/)


## solution

```python
class TicTacToe(object):

    def __init__(self, n):
        """
        :type n: int
        """
        self.n = n
        self.row = [0] * n
        self.col = [0] * n
        self.left = self.right = 0

    def move(self, row, col, player):
        """
        :type row: int
        :type col: int
        :type player: int
        :rtype: int
        """
        if player == 1:
            delta = 1
        else:
            delta = -1
        
        self.row[row] += delta
        self.col[col] += delta
        if row == col:
            self.left += delta
        if row == self.n - 1 - col:
            self.right += delta

        if abs(self.row[row]) == self.n or abs(self.col[col]) == self.n or abs(self.left) == self.n or abs(self.right) == self.n:
            return player
        
        return 0
```
时间复杂度：O() <br>
空间复杂度：O()
