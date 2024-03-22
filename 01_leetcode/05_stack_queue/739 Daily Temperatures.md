# 739 Daily Temperatures
[https://leetcode.com/problems/daily-temperatures/](https://leetcode.com/problems/daily-temperatures/)


## solution
- 暴力
```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = []
        for i in range(len(temperatures)):
            greater = False
            for j in range(i+1, len(temperatures)):
                if temperatures[j] > temperatures[i]:
                    res.append(j-i)
                    greater = True
                    break
            if not greater:
                res.append(0)
        return res
```
时间复杂度：O(n^2) <br>
空间复杂度：O()

- 单调栈
```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)  # 初始化一个默认值, 当不更新为自己, 因此初始化0
        stack = [0]  # 递增栈，对于第一个元素的判断，初始化一个比所有元素都小的

        for i in range(1, len(temperatures)):
            if temperatures[i] <= temperatures[stack[-1]]:
                stack.append(i)
            else:  # 注意 while
                while len(stack) != 0 and temperatures[i] > temperatures[stack[-1]]:
                    res[stack[-1]] = i - stack[-1]
                    stack.pop()
                stack.append(i)
        return res
```

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []  # 单调递增栈，小的pop出来
        for j, temp in enumerate(temperatures):
            while stack and temp > temperatures[stack[-1]]:
                i = stack.pop()
                res[i] = j - i
            stack.append(j)
        return res
```
时间复杂度：O(n) <br>
空间复杂度：O(n)


## follow up
[84 Largest Rectangle in Histogram](./84%20Largest%20Rectangle%20in%20Histogram.md)
