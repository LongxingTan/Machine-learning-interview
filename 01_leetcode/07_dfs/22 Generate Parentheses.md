# 22 Generate Parentheses
[https://leetcode.com/problems/generate-parentheses/](https://leetcode.com/problems/generate-parentheses/)


## solution

- dfs
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        path = []
        res = []
        self.dfs(0, 0, n, path, res)
        return res

    def dfs(self, left: int, right:int, n: int, path: List, res: List):
        if len(path) == 2 * n:
            res.append(''.join(path))
            return

        if left < n:  # 左右条件是关键
            path.append('(')
            self.dfs(left + 1, right, n , path, res)
            path.pop()

        if right < left:
            path.append(')')
            self.dfs(left, right + 1, n , path, res)
            path.pop()
```
时间复杂度：O(4^n / sqrt(n)) <br>
空间复杂度：O(n)


## follow op

[括号类小结](../05_stack_queue/20.%20Valid%20Parentheses.md)
