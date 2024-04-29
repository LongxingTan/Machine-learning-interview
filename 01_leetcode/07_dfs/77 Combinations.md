# 77 Combinations
[https://leetcode.com/problems/combinations/](https://leetcode.com/problems/combinations/)


## solution

回溯
- 整体输出结果、单次尝试的path、本次选择的选项
- 本题回溯时，注意不需要重复项，通过控制**开始index**不从前面选
  - index: 下一层for循环搜索的起始位置

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        path = []
        self.dfs(res, path, n, k, start=1)
        return res

    def dfs(self, res, path, n, k, start):
        if len(path) == k:
            res.append(path[:])  # [:]和下面的copy保留一个即可, python中一定要有
            return
        
        for i in range(start, n+1):  # 本层集合中的元素，递归N叉树，树节点孩子数量就是集合的大小         
            path.append(i)
            self.dfs(res, path.copy(), n, k, i+1)
            path.pop()
```
时间复杂度：O(n * 2^n) <br>
空间复杂度：O(n)
