# 403 Frog Jump
[https://leetcode.com/problems/frog-jump/](https://leetcode.com/problems/frog-jump/)


## solution

```python
class Solution(object):
    def canCross(self, stones):
        n = len(stones)
        stoneSet = set(stones)
        visited = set()
        def goFurther(value,units):
            if (value+units not in stoneSet) or ((value,units) in visited):
                return False
            if value+units == stones[n-1]:
                return True
            visited.add((value,units))
            return goFurther(value+units,units) or goFurther(value+units,units-1) or goFurther(value+units,units+1)
        return goFurther(stones[0],1)
```
时间复杂度：O() <br>
空间复杂度：O()


- 动态规划
```python
# dp[i][k] = dp[j][k - 1] || dp[j][k] || dp[j][k + 1]

```
