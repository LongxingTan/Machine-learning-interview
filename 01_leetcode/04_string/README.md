# 字符串

- 很多字符串的解法是two pointer， 是DP（子串问题）或者DFS
  - LIS（最长递增子序列）
  - LCS（最长公共子序列、最长公共子串）
  - LCP（最长公共前缀）
  - LPS（最长回文子序列、最长回文子串）
  - ED（最小编辑距离，也叫 “Levenshtein 距离”）、
  - KMP（一种字符串匹配的高效算法）
- palindrome
  - 可以考虑reverse string来比较
  - 找palindrome可以expand from center，或者DP
- 用特殊符号的时候，不要用-， 以防是数字转换过来的，负数也带-

- 判断非空
```python
if substring:
  pass
```

- 一个或多个空格的分割
```python
string.split()
```

- ASCII
```python
ord('a')
```


## reference
- [字符串匹配算法KMP小结](https://www.cnblogs.com/grandyang/p/6992403.html)
