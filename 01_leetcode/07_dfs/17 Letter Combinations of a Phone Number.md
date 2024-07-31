# 17 Letter Combinations of a Phone Number
[https://leetcode.com/problems/letter-combinations-of-a-phone-number/](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)


## solution

- 回溯
```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        self.maps = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }

        path = []
        res = []
        index = 0
        self.dfs(path, res, digits, index)
        return res

    def dfs(self, path, res, digits, index):
        if len(path) == len(digits):  # 或者 index == len(digits)
            res.append("".join(path))
            return

        i = digits[index]
        letters = self.maps[i]

        for letter in letters:
            path.append(letter)
            self.dfs(path, res, digits, index+1)
            path.pop()
```
时间复杂度：O(4^n) <br>
空间复杂度：O(4^n)
