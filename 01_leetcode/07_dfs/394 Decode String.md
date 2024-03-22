# 394 Decode String
[https://leetcode.com/problems/decode-string/description/](https://leetcode.com/problems/decode-string/description/)


## solution

- 栈
```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        cur_num = 0
        cur_str = ''

        for char in s:
            if char == '[':
                stack.append(cur_str)  # 栈也可以直接保存元素 (cur_str, cur_num)
                stack.append(cur_num)
                cur_str = ''
                cur_num = 0
            elif char == ']':
                num = stack.pop()
                prev_str = stack.pop()
                cur_str = prev_str + num * cur_str
            elif char.isdigit():
                cur_num = cur_num * 10 + int(char)
            else:
                cur_str += char
        return cur_str
```
时间复杂度：O() <br>
空间复杂度：O()
