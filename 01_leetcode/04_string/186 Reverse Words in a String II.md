# 186 Reverse Words in a String II
[https://leetcode.com/problems/reverse-words-in-a-string-ii/](https://leetcode.com/problems/reverse-words-in-a-string-ii/)


## solution

```python
class Solution(object):
    def reverseWords(self, s):
        self.reverse_part(s, 0, len(s) - 1)
        p = 0
        for index in range(1, len(s)):
            if s[index] == ' ':
                self.reverse_part(s, p, index - 1)
                p = index + 1
        
        self.reverse_part(s, p, len(s) - 1)
    
    def reverse_part(self, s, left, right):
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
```
时间复杂度：O(n) <br>
空间复杂度：O(1)
