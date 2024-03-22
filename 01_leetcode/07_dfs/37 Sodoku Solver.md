# 37 Sodoku Solver
[https://leetcode.com/problems/sudoku-solver/](https://leetcode.com/problems/sudoku-solver/)


## solution

```python

```
时间复杂度：O() <br>
空间复杂度：O()


## follow up

[36. Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)

- 同一行，同一列，同一小方形不能有重复值

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        res = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == '.':
                    continue
                
                element = board[i][j]
                res += [(i, element), (element, j), (i//3, j//3, element)]
        return len(res) == len(set(res))
```
时间复杂度：O() <br>
空间复杂度：O()


```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        s = set()
    
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    c = board[i][j]
    
                    # Row check
                    key = f'{c} in row {i}' # same as -> str(c) + ' in row ' + str(i)
    
                    if key in s:
                        return False
                    else:
                        s.add(key)
    
                    # Column check
                    key = f'{c} in col {j}' # same as -> str(c) + ' in col ' + str(j)
    
                    if key in s:
                        return False
                    else:
                        s.add(key)
    
                    # Box check
                    boxIndex = (i // 3) * 3 + (j // 3)
                    key = f'{c} in box {boxIndex}' # same as -> str(c) + ' in box ' + str(boxIndex)
    
                    if key in s:
                        return False
                    else:
                        s.add(key)     
        return True
```
