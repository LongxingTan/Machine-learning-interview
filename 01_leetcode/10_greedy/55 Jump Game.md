# 55 Jump Game
[https://leetcode.com/problems/jump-game/](https://leetcode.com/problems/jump-game/)


## solution

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        
        cover = 0  # 含义是目前能覆盖的最大范围

        for i in range(len(nums)):
            if i <= cover:  # 先要确保目前这一步能到
                cover = max(cover, i+nums[i])
                if cover >= len(nums) - 1:
                    return True
        return False
```

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if not nums:
            return False
        if len(nums) == 1:
            return True

        i = 0
        res = 0        
       
        while i <= res:
            res = max(res, i + nums[i])
            if res >= len(nums) - 1:
                return True    
            i += 1     

        return False
```
时间复杂度：O() <br>
空间复杂度：O()


## follow up

[45 Jump Game II](https://leetcode.com/problems/jump-game-ii/)

- 记录当前覆盖最远距离和下一次覆盖最远距离，当前距离最远时，增加一步到下一次最远距离
```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0

        cur_pos = 0
        next_pos = 0  
        res = 0
        for i in range(len(nums)):
            next_pos = max(i+nums[i], next_pos)
            if i == cur_pos:
                cur_pos = next_pos
                res += 1
                if next_pos >= len(nums) - 1:
                    break
        return res
```
时间复杂度：O() <br>
空间复杂度：O()

- 动态规划
```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        res = [float('inf')] * len(nums)
        res[0] = 0

        for i in range(len(nums)):
            for j in range(i, i+nums[i]+1):
                if j < len(nums):  # 注意边界
                    res[j] = min(res[j], res[i]+1)  # 递推是根据业务含义来的
        return res[-1]
```
时间复杂度：O() <br>
空间复杂度：O()


[1340. Jump Game V](https://leetcode.com/problems/jump-game-v/description/)
```python

```


[1871. Jump Game VII](https://leetcode.com/problems/jump-game-vii/description/)
```python
# 滑动窗口+DP: https://zhuanlan.zhihu.com/p/474961276

```

- 超时
```python
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        if s[-1] != '0':
            return False
        
        nums = [0]
        for i, char in enumerate(s[1:]):
            if char == '0':
                nums.append(i+1)  
        
        dp = [False] * len(nums)        
        dp[0] = True
        
        for i in range(1, len(nums)):
            for j in range(i):
                if dp[j] and minJump <= nums[i] - nums[j] <= maxJump:
                    dp[i] = True
                    break   
        return dp[-1]        
```
