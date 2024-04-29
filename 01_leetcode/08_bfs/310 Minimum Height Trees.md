# 310 Minimum Height Trees
[https://leetcode.com/problems/minimum-height-trees/](https://leetcode.com/problems/minimum-height-trees/)


## solution

- 问题转化为：找出距离其他所有节点（特别是叶节点）最近的节点，在树状图中寻找所有质心节点
- 利用拓扑排序：从较少degree的节点往上游

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        graph = collections.defaultdict(list)
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        
        leaves = [x for x in range(n) if len(graph[x]) <= 1]
       
        while len(graph) > 2:
            new_leaves = []
            for leaf in leaves:                
                neighbor = graph[leaf].pop()
                del graph[leaf]
                graph[neighbor].remove(leaf)
                if len(graph[neighbor]) == 1:
                    new_leaves.append(neighbor)
            leaves = new_leaves
        return leaves
```
时间复杂度：O() <br>
空间复杂度：O()


- 超时BFS
```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        graph = collections.defaultdict(list)
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        
        res = {}
        for i in range(n):
            depth = self.bfs(i, graph, n)
            res.update({i: depth})        

        min_depth = min(res.values())
        ans = []
        for i, j in res.items():
            if j == min_depth:
                ans.append(i)
        return ans
    
    def bfs(self, start, graph, n):
        queue = collections.deque([start])        
        visited = [False] * n
        visited[start] = True
        step = 0
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                for adj in graph[node]:
                    if not visited[adj]:
                        visited[adj] = True
                        queue.append(adj)
            step += 1
        return step
```