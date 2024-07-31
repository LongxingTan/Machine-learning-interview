# 广度优先搜索

常见的BFS用来解决什么问题？
- 简单树/图（有向无向皆可）的最短路径长度，注意是长度而不是具体的路径
- 拓扑排序
  - 一般用于解决有方向的图，non-directed graph不能用， 需要用union find或者普通的DFS/BFS
- 遍历一个图（或者树）
- use BFS for unweighted graphs and use Dijkstra's algorithm for weighted graphs. Dijkstra is similar to BFS, the difference is using priority queue instead of queue. The priority will pop the element with higher priority
- 复杂度
  - The time complexity will be O(V+E), each vertex will become a source only once and each edge will be accessed and removed once.
  - The space complexity will be O(V+E), since we are storing all of the edges for each vertex in an adjacency list


**基础**
- BFS基本模板（需要记录层数或者不需要记录层数）
- BFS和DFS都是一种N叉树的遍历，图有必要的时间记录是否被visited。**加入队列的时候**，立刻标记为被访问过
- 多数情况下时间复杂度空间复杂度都是O（N+M），N为节点个数，M为边的个数
- 有些题目既可以DFS也可以BFS，此时"用BFS更好还是DFS更好取决于树的形状是又高又窄还是又矮又宽"


## 一些trick
- 通过去掉leave nodes来实现图的bfs


## 基于树的BFS
不需要专门一个set来记录访问过的节点

```python
bfs = [target.val]
visited = set([target.val])
for k in range(K):
    bfs = [y for x in bfs for y in conn[x] if y not in visited]
    visited |= set(bfs)
```

## 基于图的bfs
- [207 Course Schedule](./207%20Course%20Schedule.md)
- [210. Course Schedule II](./210.%20Course%20Schedule%20II.md)

```python
import collections

collections.defaultdict(list)
collections.defaultdict(set)
collections.defaultdict(dict)
collections.deque()
```

[1971. Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/description/)
```python
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        if not edges:
            return True

        graph = collections.defaultdict(list)
        visited = set()
        for (i, j) in edges:
            graph[i].append(j)  # 无向图两个方向都加入
            graph[j].append(i)

        nodes = collections.deque([source])
        visited.add(source)

        while nodes:
            i = nodes.popleft()
            i_next = graph[i]
            for node in i_next:
                if node == destination:
                    return True
                if node not in visited:
                    nodes.append(node)
                    visited.add(node)
        return False
```

## follow up
- graph类的BFS如何scale
- [BFS｜宽度优先搜索 题型技巧分类总结](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=909366&ctid=9)
- [How to trace the path in a Breadth-First Search?](https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search)
- 如何打印最短路径/步数
  - python中，直接把path也传到queue里面，每次更新path
