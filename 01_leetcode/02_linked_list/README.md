# 链表

```python
class ListNode():
    self.val = val
    self.next = next    
```

## 基础
- 单链表
- 双链表
- 循环链表

与列表的区别
- 列表Array的增和删复杂度O(n),查找O(1)
- 链表List的增和删O(1)，查找O(n)


## 技巧
链表反转和快慢指针几乎是所有链表类问题的基础

- 头部加dummy node，便于返回整个链表头节点
```python
dummy = ListNode(-1)
dummy.next = head
```

- 插入节点
```python
new = ListNode(val=0)
new.next = cur.next  # 适当的先后顺序
cur.next = new
```

- 删除节点
```python
cur.next = cur.next.next
```
