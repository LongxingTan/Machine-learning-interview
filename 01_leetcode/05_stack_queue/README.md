# 栈与队列

- pop前注意检查是不是为空
 

## 栈
> 栈的一些题目思路并不容易，比如计算器224/227，decoder string 394

[Design a stack in python](https://www.geeksforgeeks.org/design-a-stack-that-supports-getmin-in-o1-time-and-o1-extra-space/)

[stack in java](https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/Stack.java.html)

[Implementing Stacks- From Basic to more advanced concepts](https://machine-learning-made-simple.medium.com/implementing-stacks-from-basic-to-more-advanced-concepts-98fb06924936)


**Vanilla Stacks**
```python
class Stack:
    def __init__(self):
        self.items = []  
    
    def push(self, item):
        """Add an item to the top of the stack."""
        self.items.append(item)
    
    def pop(self):
        """Remove and return the top item of the stack."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self.items.pop()
    
    def peek(self):
        """Return the top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.items[-1]
    
    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.items) == 0
```


**Using a Singly Linked List**
```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None        
        
class LinkedListStack:
    def __init__(self):
        self.top = None
        self.size = 0
    
    def push(self, value):
        new_node = Node(value)
        new_node.next = self.top
        self.top = new_node
        self.size += 1
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        value = self.top.value
        self.top = self.top.next
        self.size -= 1
        return value
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.top.value
    
    def is_empty(self):
        return self.size == 0
```

**Persistent Stack**
```python
class PersistentStack:
    def __init__(self):
        self.versions = [[]]
        self.current_version = 0
    
    def push(self, item):
        new_version = self.versions[self.current_version].copy()
        new_version.append(item)
        self.versions.append(new_version)
        self.current_version += 1
    
    def pop(self):
        if self.current_version == 0:
            raise IndexError("Pop from an empty stack")
        new_version = self.versions[self.current_version].copy()
        new_version.pop()
        self.versions.append(new_version)
        self.current_version += 1
```

**Thread Safety**
```python
import threading

class ThreadSafeStack(Stack):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
    
    def push(self, item):
        with self.lock:
            super().push(item)
    
    def pop(self):
        with self.lock:
            return super().pop()
```


## 单调栈 monotone stack
单调栈通常是一维数组，用于解决数组中找出每个数字左右/右边第一个大于／小于该数字的位置或者数字；单调的意思是保留在栈或者队列中的数字是单调递增或者单调递减的

- 直观感受的话，每次遇到比之前大的元素才更新之前的，否则暂时存入栈
- 比如找右边比自己大的元素，如果右边元素太小，该元素先在栈里等待直到遇到大的；遇到了同时更新结果. 遇到了就直接出栈，因此得到的是第一个比自己大的
- 单调栈的维护是 O(n) 级的时间复杂度，因为所有元素只会进入栈一次，并且出栈后再也不会进栈了


## 队列
[Introduction to Circular Queue](https://www.geeksforgeeks.org/introduction-to-circular-queue/)
