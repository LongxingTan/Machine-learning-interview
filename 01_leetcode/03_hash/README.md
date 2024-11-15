# Hashmap/ Hashset
[Implementation of Hash Table in Python](https://www.geeksforgeeks.org/implementation-of-hash-table-in-python-using-separate-chaining/)


## 基础
- O(n) 空间复杂度, 实现 近似 O(1) 时间复杂度的插入、查找、删除等
- 哈希碰撞：拉链法和线性探测法
- defaultdict: if a key is not found in the dictionary, then instead of a KeyError being thrown, a new entry is create
- C++ 中的哈希集合为 unordered_set，可以查找元素是否在集合中。如果需要同时存储键 和值，则需要用 unordered_map


## 代码
```python
somedict = {}
print(somedict[3]) # KeyError

someddict = defaultdict(int)
print(someddict[3]) # print int(), thus 0

d = defaultdict(list)
print(d[2])  # []

d = collections.defaultdict(set)
print(d[3])

# 带权重的图
graph = collections.defaultdict(dict)
```


- 排序
```python
# 根据values, 注意加items, 以及x[1]
sorted(footballers_goals.items(), key=lambda x: x[1])
```


- 增/改/删
```python
mydict.update()

# 删除字典中的某键
del mydict[key]

# 没有key时取得0
mydict[key] = mydict.get(key, 0) + 1

# 
mydict.setdefault()
```
