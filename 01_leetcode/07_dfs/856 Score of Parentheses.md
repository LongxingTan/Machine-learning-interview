# 856 Score of Parentheses
[https://leetcode.com/problems/score-of-parentheses/](https://leetcode.com/problems/score-of-parentheses/)


## solution

```python
# 左括号更新深度，右括号更新结果

class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        score = 0
        depth = 0

        for i, char in enumerate(s):
            if char == '(':
                depth += 1
            else:
                depth -= 1
                if s[i-1] == '(':
                    score += 2 ** depth
        return score
```
时间复杂度：O() <br>
空间复杂度：O()
