# 扫描线

- 不需要暴力遍历每一个点，只检测起点和重点的位置
- 有些题目和线段树结合

[lintcode-391 · 数飞机](https://www.lintcode.com/problem/391/)

```python
# 本题也可以使用前缀和

class Solution:
    def countOfAirplanes(self, airplanes):
        room=[]
        for i in airplanes:
            room.append((i.start, 1))
            room.append((i.end, -1))

        tmp = 0
        ans = 0
        room = sorted(room)
        for idx, cost in room:
            tmp += cost
            ans = max(ans,tmp)
        return ans
```

时间复杂度：O(nlog(n)) <br>
空间复杂度：O(time)

## Reference

- [一文读懂扫描线算法 - 小水的文章 - 知乎](https://zhuanlan.zhihu.com/p/103616664)
