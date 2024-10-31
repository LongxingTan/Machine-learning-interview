# 1143 Longest Common Subsequence
[https://leetcode.com/problems/longest-common-subsequence/](https://leetcode.com/problems/longest-common-subsequence/)


## solution

- 字符串头部追加一个空格，以减少边界判断/初始化
- 除了追加空格外， f[i][j]代表s1的前 i-1 个字符、s2的前 j-1 的字符

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)

        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
```
时间复杂度：O() <br>
空间复杂度：O()


## follow up

- 注意理解和[718. 最长重复子数组](./718%20Maximum%20Length%20of%20Repeated%20Subarray.md)的区别
  - 718中dp[i][j]含义是*以i-1结尾*和j-1结尾的最长重复子数组，如果值不一样，dp仍然是初始值
  - 本题中dp[i][j]含义是长度为[0, i - 1]的字符串text1与长度为[0, j - 1]的字符串text2的最长公共子序列为dp[i][j]
  - 总结下来，什么时候dp需要以i/i-1为结尾？根据意义，判断能否转移。需要以结尾的话，状态转移过程中有中断和重新开始


[583. Delete Operation for Two Strings](https://leetcode.com/problems/delete-operation-for-two-strings/)
```python

```
时间复杂度：O() <br>
空间复杂度：O()


[1035. Uncrossed Lines](https://leetcode.com/problems/uncrossed-lines/description/)
```python
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        m = len(nums1)
        n = len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
```
时间复杂度：O() <br>
空间复杂度：O()
