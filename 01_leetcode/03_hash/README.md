# Hashmap/ Hashset

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
sorted(footballers_goals.items(), key=lambda x:x[1])
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


- 实现
```python
# https://www.geeksforgeeks.org/implementation-of-hash-table-in-python-using-separate-chaining/
class Node: 
    def __init__(self, key, value): 
        self.key = key 
        self.value = value 
        self.next = None
  
  
class HashTable: 
    def __init__(self, capacity): 
        self.capacity = capacity 
        self.size = 0
        self.table = [None] * capacity 
  
    def _hash(self, key): 
        return hash(key) % self.capacity 
  
    def insert(self, key, value): 
        index = self._hash(key) 
  
        if self.table[index] is None: 
            self.table[index] = Node(key, value) 
            self.size += 1
        else: 
            current = self.table[index] 
            while current: 
                if current.key == key: 
                    current.value = value 
                    return
                current = current.next
            new_node = Node(key, value) 
            new_node.next = self.table[index] 
            self.table[index] = new_node 
            self.size += 1
  
    def search(self, key): 
        index = self._hash(key) 
  
        current = self.table[index] 
        while current: 
            if current.key == key: 
                return current.value 
            current = current.next
  
        raise KeyError(key) 
  
    def remove(self, key): 
        index = self._hash(key) 
  
        previous = None
        current = self.table[index] 
  
        while current: 
            if current.key == key: 
                if previous: 
                    previous.next = current.next
                else: 
                    self.table[index] = current.next
                self.size -= 1
                return
            previous = current 
            current = current.next
  
        raise KeyError(key) 
  
    def __len__(self): 
        return self.size 
  
    def __contains__(self, key): 
        try: 
            self.search(key) 
            return True
        except KeyError: 
            return False
```