# 列表

- 二次遍历
  - 第一次遍历得到某辅助指标，第二次利用该指标完成任务
- 相邻比较
- 互换


## Python常用操作

- 从列表list中找出目标值 **第一个匹配项的索引**
```python
a = 0
list.index(a)
```

- python中使用列表实现队列

```python
import collections

q = list.pop(0)  # index=0, pop后，q=list[0], list=[剩余元素]
# list.remove(element)
# list.append(element)

q = collections.deque()
q.popleft()
q.pop()  # 从right pop
q.append(1)
q.insert(0, 1)
q.extend(list)  # 两个list concat, 不返回
```

- 往某个位置插入某个元素
```python
# 手工翻转一个list通过不断往0位置插入
# 时间复杂度是 O(n)，插入操作需要将列表中所有现有的元素向右移动一个位置，以便为新元素腾出空间
list.insert(0, num)
```

- 列表排序
```python
l = sorted(l)
l = sorted(l, key = lambda x: x[0])
l = sorted(l, reverse=True)

l.sort()  # 注意这里是in-place

# 多个key
array.sort(key=lambda element: (element[1], element[2]), reverse=True) # 多个sorting key

# 双list排 或者直接 循环时排
[x for _, x in sorted(zip(Y, X), key=lambda pair: pair[0])]

# 自定义
cars = ['Ford', 'Mitsubishi', 'BMW', 'VW']

def myFunc(e):
  return len(e)

cars.sort(reverse=True, key=myFunc)
```

- 索引倒序遍历
```python
range(len(nums) - 1, -1, -1)
```

- 列表索引
```python
# 左闭右开
# 从右边取，注意为0会取全体值是否是想要的
list[-1:]  # 取最后一个
list[-0:]  # 取全体list
list[::2]  # 每间隔2个取
```

- deque
```python
# BFS常用
# my_deque.reverse() 不会创建新的内存空间
from collections import deque

queue = deque([1,2,3])
queue.append(4)
queue.popleft()
```

- 定义二维列表
```python
matrix = [[0 for _ in range(col)] for _ in range(row)]
matrix1 = [[0] * col for _ in range(row) ]

# 错误，都是引用导致值相同
[[0] * col] * row
```

- 二维列表转置
```python

```


- 个位是否为1
```python
x % 10 == 1
```

- heapq 堆的操作
```python
import heapq

data = [2,3,5,1]
heapq.heapify(data)
heapq.heappush(data, 6)
print(heapq.heappop(data))
```

- 链表循环内容涉及两个节点判断时
```python
# 根据head / head.next / head & head.next 来决定迭代终点
while head is not None and head.next is not None:
    pass
```

- 字符串前的空格
```python
s.strip()
```

- 字符串
```python
# ord
ord(s[0])/ s[0].isdigit()

# 小写
s.lower()
```

- 两层循环中的break，break的是相应层的循环


- 多个条件判断时，条件是有先后顺序的。是否越界和是否符合条件可以写在一起
```python
# 正确
while s[l] == s[r] and l >= 0 and r < len(s):

# 错误
while l >= 0 and r < len(s) and s[l] == s[r]:
```

- for循环：提前已知遍历的所有元素；while循环：根据循环过程来决定所有遍历元素

- 最小与最大

```python
float('inf')
float('-inf')
```

- flag来交替
```python
# 280. Wiggle Sort

```
