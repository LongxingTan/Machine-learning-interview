# Hashmap/ Hashset
- [Implementation of Hash Table in Python](https://www.geeksforgeeks.org/implementation-of-hash-table-in-python-using-separate-chaining/)
- [Python Hash Sets Explained & Demonstrated - Computerphile](https://www.youtube.com/watch?v=9oKpRTBfNXo)


## 基础
- O(n) 空间复杂度, 实现 近似 O(1) 时间复杂度的插入、查找、删除等
- hash(key) % capacity, Hash 函数的主要作用是将任意大小的数据映射到一个固定大小的索引
- 哈希碰撞：拉链法和线性探测法
- defaultdict: if a key is not found in the dictionary, then instead of a KeyError being thrown, a new entry is created
- C++ 中的哈希集合为 unordered_set，可以查找元素是否在集合中。如果需要同时存储键 和值，则需要用 unordered_map


## 代码

```python
mydict = {}
print(mydict[3]) # KeyError

mydict = collections.defaultdict(int)
print(mydict[3]) # print int(), thus 0

d = collections.defaultdict(list)
print(d[2])  # []

d = collections.defaultdict(set)
print(d[3])

# 带权重的图
graph = collections.defaultdict(dict)
```


- 字典按value排序
```python
dict(sorted(mydict.items(), key=lambda x: x[1]))
# 注意如果返回sorted, 返回的是list[tuple], 如[(1, 3), (2, 2), (3, 1)]
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




- 字典获得key值
```python
mydict.get(key, default)
```

- 删除键值
```python
del mydict[key]

value = mydict.pop(key, None)
```

- 键或值转化为array
```python
list(mydict.keys())[:k]

list(mydict.values())[:k]
```
