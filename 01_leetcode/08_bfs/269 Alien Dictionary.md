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
                if pre != next:  # 前后两个单词第一个不一样的字母
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


## follow up

[953. Verifying an Alien Dictionary](https://leetcode.com/problems/verifying-an-alien-dictionary/description/)
```python
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        mydict = {v:i for i, v in enumerate(order)}
        words_order = []
        for word in words:
            words_order.append([mydict[char] for char in word])

        # for w1, w2 in zip(words, words[1:]):          
        #     if w1 > w2:
        #         return False
        
        return all(w1 <= w2 for w1, w2 in zip(words_order, words_order[1:]))
```
