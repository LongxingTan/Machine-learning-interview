# 269 Alien Dictionary
[https://leetcode.com/problems/alien-dictionary/](https://leetcode.com/problems/alien-dictionary/)


## solution

- 把字母的大小关系转换为有向图, 用拓扑排序解决

```python
class Solution(object):
    def alienOrder(self, words):
        graph = {}
        # A list stores No. of incoming edges of each vertex
        in_degree = [0] * 26
        self.build_graph(graph, words, in_degree)
        return self.topology_sort(graph, in_degree)
    
    def build_graph(self, graph, words, in_degree):
        pass
    
    def topology_sort(self, graph: Dict[chr, Set[chr]], inDegrees: List[int]):
        pass        
```
时间复杂度：O() <br>
空间复杂度：O()
