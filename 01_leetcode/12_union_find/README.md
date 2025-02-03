# 并查集 Union find

- 某些题可以用DFS/BFS，DFS/BFS可能会有找不到某些点，只能用union find
- 并查集常用来解决连通性问题，当需要判断两个元素是否在同一个集合里时，想到用并查集
- 用集合中的一个元素代表集合


```python
class DSU:
    def __init__(self):
        self.parent = range(10001)  # 用一个数组来存储每个元素的父节点，初始时每个节点parent都是自己

    def find(self, x):  # 查询parent
        if x != self.parent[x]:  # 注意是if
            self.parent[x] = self.find(self.parent[x])  # 一层一层访问父节点，直至根节点
        return self.parent[x]  # 返回父节点
    
    def union(self, x, y):  # 合并
        self.parent[self.find(x)] = self.find(y)  # 先找到两个集合的代表元素，然后将前者的父节点设为后者
    
    def same(self, x, y):
        return self.find(x) == self.find(y)
```


## 优化
- 路径压缩
  - find过程，把沿途的每个节点的父节点都设为根节点
- 按秩合并
  - 把简单的树往复杂的树上合并; 合并后，到根节点距离变长的节点个数比较少


## 阅读
- [算法学习笔记(1) : 并查集](https://zhuanlan.zhihu.com/p/93647900)
