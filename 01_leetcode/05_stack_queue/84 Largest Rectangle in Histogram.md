# 84 Largest Rectangle in Histogram
[https://leetcode.com/problems/largest-rectangle-in-histogram/](https://leetcode.com/problems/largest-rectangle-in-histogram/)


## solution

- 单调栈: 找每个柱子左右两边第一个小于该柱子的柱子

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        stack = []  # 单调栈
        next_smaller_left = [0] * n
        for i in range(n):
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            if stack:
                next_smaller_left[i] = stack[-1] + 1
            stack.append(i)
        
        stack = []
        next_smaller_right = [n - 1] * n
        for i in range(n-1, -1, -1):
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            if stack:
                next_smaller_right[i] = stack[-1] - 1
            stack.append(i)
        
        res = heights[0]
        for i in range(n):
            height = heights[i]
            width = next_smaller_right[i] - next_smaller_left[i] + 1
            area = height * width
            res = max(res, area)
        return res
```
时间复杂度：O() <br>
空间复杂度：O()


## follow up

[42 接雨水](../01_two_pointers/42.%20Trapping%20Rain%20Water.md)
