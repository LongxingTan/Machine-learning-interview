# 269 Alien Dictionary
[https://leetcode.com/problems/alien-dictionary/](https://leetcode.com/problems/alien-dictionary/)


## solution

- 把字母的大小关系转换为有向图, 用拓扑排序解决

```python
from heapq import *

class Solution:
    def alienOrder(self, words):
        # Construct Graph
        in_degree = {char: 0 for word in words for char in word}
        neighbors = collections.defaultdict(list)
        for pos in range(len(words) - 1):
            for i in range(min(len(words[pos]), len(words[pos+1]))):
                pre, next = words[pos][i], words[pos+1][i]
                if pre != next:
                    in_degree[next] += 1
                    neighbors[pre].append(next)
                    break
        
        # Topological Sort
        heap = [ch for ch in in_degree if in_degree[ch] == 0]
        heapify(heap)
        order = []
        while heap:
            for _ in range(len(heap)):
                ch = heappop(heap)
                order.append(ch)
                for child in neighbors[ch]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        heappush(heap, child)
        
        # order is invalid
        if len(order) != len(in_degree):
            return ""
        return ''.join(order)     
```
时间复杂度：O(n * l) <br>
空间复杂度：O(26)
