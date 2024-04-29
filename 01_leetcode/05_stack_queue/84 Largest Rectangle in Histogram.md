# 84 Largest Rectangle in Histogram
[https://leetcode.com/problems/largest-rectangle-in-histogram/](https://leetcode.com/problems/largest-rectangle-in-histogram/)


## solution

- 单调栈: 左边第一个比自己小的index, 构成矩形

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 输入数组首尾各补上一个0
        heights.insert(0, 0)
        heights.append(0)
        stack = [0]
        result = 0
        for i in range(1, len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                mid_height = heights[stack[-1]]
                stack.pop()
                if stack:
                    # area = width * height
                    area = (i - stack[-1] - 1) * mid_height
                    result = max(area, result)
            stack.append(i)
        return result
```

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


- 双指针: 找每个柱子左右两边第一个小于该柱子的柱子
```python

```


## follow up

[42 接雨水](../01_two_pointers/42.%20Trapping%20Rain%20Water.md)
