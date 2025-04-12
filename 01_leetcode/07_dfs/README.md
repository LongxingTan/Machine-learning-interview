# 深度优先搜索

DFS要有**子问题**的思路：子问题是什么？子问题如何状态转移到整体问题。通过把实例数据的方法，画出解决问题的多叉树，从中归纳出子问题与子问题的转移

主函数用于遍历所有的搜索位置，判断是否可以开始搜索。辅函数则负责具体的深度优先搜索的递归调用

**搜索类**

- 树、图等节点
- 序列等位于前面的关系

- 一维 or 二维(二叉树) dfs
- 记录状态的dfs (图)

基于图的DFS: 和BFS一样一般需要一个set来记录访问过的节点，避免重复访问造成死循环; Word XXX 系列面试中非常常见，例如word break，word ladder，word pattern，word search

基于排列组合的DFS: 其实与图类DFS方法一致，但是排列组合的特征更明显

记忆化搜索（DFS + Memoization Search）：算是用递归的方式实现动态规划，递归每次返回时同时记录下已访问过的节点特征，避免重复访问同一个节点，可以有效的把指数级别的DFS时间复杂度降为多项式级别; 注意这一类的DFS必须在最后有返回值（分治法），不可以用回溯法; for循环的dp题目都可以用记忆化搜索的方式写，但是不是所有的记忆化搜索题目都可以用for循环的dp方式写。

## 二叉树

- [基础](https://stackoverflow.com/questions/2598437/how-to-implement-a-binary-tree)
  - 注意一定有递归(recursion)终止条件，如果是叶子节点下面的空，可以直接返回；如果是中间几点，可以中途根据不同条件判断来return。如果有返回值的递归函数，可能是需要对节点返回值进行处理，也就是由子问题状态转移到整体问题上
  - 注意如何在递归函数中不同阶段进行返回
- 遍历 traversal
  - preorder/ inorder/ postorder/ level
    - 递归转迭代: preorder 直接用 stack; inorder 用 stack + cur; postorder 用 stack + cur + prev;
  - 递归写法与非递归/栈写法
  - [了解]空间优化 Morris traversal: iterative approach to leverage preorder traversal that the entire left subtree would be placed between the node and its right subtree
- 属性
  - 检查子树结构的题都需要一个 helper 函数，输入带两个 root
- 构造
- 二叉搜索树
  - 中序遍历是从小到大
  - 一些问题可以从排序列表的角度先得到思路，再转化到BST中
  - 除了中序遍历，常用还有分治法（搜索一半树或搜索整个树）
- divide & conquer
  - 几种方式
- 如果tree/graph node不存在本地怎么办
  - 把graph的attributes都改成RPC

```text
# 一: 搜索一条边
# 注意这里返回，可以直接返回，也可以返回其子树的结果。返回子树的效果相当于本节点被忽略，适合BST按一定范围的题目
if (递归函数(root->left)) return ;
if (递归函数(root->right)) return ;
return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)


# 二: 搜索整颗树
left = 递归函数(root->left);
right = 递归函数(root->right);
left与right的逻辑处理;
return
```

## 回溯

回溯的本质仍然是穷举所有可能，需要记录节点状态的深度优先搜索，可以返回所有解。

- N叉树
  - 树的深度为总的选项个数
  - 集合中递归查找子集，候选集合的大小就构成了树的宽度; 递归构成的树的深度
- 无返回
- for循环横向遍历，递归纵向遍历，回溯不断调整结果集

回溯中递归函数的项：

- paths 路径，已经做过的选择
- 选择列表，当期可以做出的选择
- res 结果列表

每个节点既有路径也有选择。

- 知道回溯算法中如果什么都不设置会输出什么结果

去重

- 如果一个元素不能重复使用，需要startIndex，调整下一层递归的起始位置
- 如果不同位置的相同元素结果相同，那么排序，在横向for循环时判断
- 如果子序列不同排序，则采用额外的数组记录是否使用过

## 图搜索

常见的DFS用来解决什么问题？

- 1 图中（有向无向皆可）的符合某种特征（比如最长）的路径以及长度
- 2 排列组合
- 3 遍历一个图（或者树）
- 4 找出图或者树中符合题目要求的全部方案

DFS基本模板（需要记录路径，不需要返回值 and 不需要记录路径，但需要记录某些特征的返回值）
除了遍历之外多数情况下时间复杂度是指数级别，一般是O(方案数×找到每个方案的时间复杂度)
递归题目都可以用非递归迭代的方法写，但一般实现起来非常麻烦
基于树的DFS：需要记住递归写前序中序后序遍历二叉树的模板

不记录visited的话，有两个思路

- 递归的时候直接改原矩阵，递归返回的时候在改回去就行了。 要求输入能够被更改，但是程序跑完之后输入还是一样的， 不会被破环。
- 直接用UnionFind，检查连通性即可

Dijkstra 是一个最短路径算法，他的核心就是边的松弛

## 记忆化搜索

> DFS + Memo就是DP

- [329 Longest Increasing Path in a Matrix](./329%20Longest%20Increasing%20Path%20in%20a%20Matrix.md)

## 代码

- DFS

```python
graph1 = {
    'A': ['B', 'S'],
    'B': ['A'],
    'C': ['D', 'E', 'F', 'S'],
    'D': ['C'],
    'E': ['C', 'H'],
    'F': ['C', 'G'],
    'G': ['F', 'S'],
    'H': ['E', 'G'],
    'S': ['A', 'C', 'G']
}


def dfs(graph, node):
    visited = [False] * len(graph)
    dfs_utils(node, visited)


def dfs_utils(graph, node, visited):
    visited[node] = True
    for i in graph[node]:
        if visited[i] == False:
            dfs_utils(graph, i, visited)


def dfs1(graph, node, visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs1(graph, n, visited)
    return visited


def dfs2(graph, node):
    visited = [node]
    stack = [node]
    while stack:
        node = stack[-1]
        if node not in visited:
            visited.extend(node)
        remove_from_stack = True
        for next in graph[node]:
            if next not in visited:
                stack.extend(next)
                remove_from_stack = False
                break
        if remove_from_stack:
            stack.pop()
    return visited

def dfs_iterative(graph, start_vertex):
    visited = set()
    traversal = []
    stack = [start_vertex]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            traversal.append(vertex)
            stack.extend(reversed(graph[vertex]))   # add vertex in the same order as visited
    return traversal

if __name__ == '__main__':
    print(dfs(graph1, 'A', []))
    print(dfs2(graph1, 'A'))
```

四个方向初始化和遍历

```python
directions = [(1,0),(-1,0),(0,1),(0,-1)]

for dir in directions:
    x, y = i + dir[0], j + dir[1]
```

DFS辅助函数

```python
def dfs(self, i, j, matrix, visited, m, n):
    if visited:
        # return or return a value

    for dir in self.directions:
        x, y = i + dir[0], j + dir[1]
        if x < 0 or x >= m or y < 0 or y >= n or matrix[x][y] <= matrix[i][j] (or a condition you want to skip this round):
            continue
        # do something like
        visited[i][j] = True
        # explore the next level like
        self.dfs(x, y, matrix, visited, m, n)
```
