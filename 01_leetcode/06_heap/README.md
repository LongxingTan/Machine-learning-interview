# Heap／Priority Queue

## 基础

- **完全二叉树** (complete binary tree)
- list[建堆复杂度](https://zhuanlan.zhihu.com/p/676546653)为O(n)，获得最大值复杂度为O(1)，pop取出最大值或插入任意值复杂度为O(log(n))，因为heapify的复杂度为log(n)
  - https://stackoverflow.com/questions/51735692/python-heapify-time-complexity
- python中只有小顶堆（min heap），最小的元素总是在根结点：heap[0]，父节点元素始终小于或等于所有子节点元素
  - 小顶堆pop时，把最小的先弹出去
- 大顶堆可以通过所有元素变为其负数
- heap中的item可以是多元素，例如包括其频次，从而记录其状态
- 求Top K/Median问题时，及时联想到heap

## 代码

```python
# 创建一个堆，可以使用list来初始化为 [] ，或者通过函数 heapify() ，把一个list转换成堆
import heapq

raw = [3, 1, 5]
heapq.heapify(raw)
print(heapq.heappop(raw))  # python把最小的pop出去
print(heapq.heappushpop(raw, -1))  # 先push 再pop
```

- find_Kth_smallest_number

```python
# 可以排序法，二分查找法，优先队列, quick select

def find_Kth_smallest_number(nums, k):
    heap = []
    for num in nums:
        if len(heap) < k:
          heappush(heap, -num)
        else:
          if -num > heap[0]:
            heappop(heap)
            heappush(heap, -num)
    return -heap[0]
```

```python
def find_Kth_smallest_number(nums, k):
    maxHeap = []
    # put first k numbers in the max heap
    for i in range(k):
        heappush(maxHeap, -nums[i])

    # go through the remaining numbers of the array, if the number from the array is smaller than the
    # top(biggest) number of the heap, remove the top number from heap and add the number from array
    for i in range(k, len(nums)):
        if -nums[i] > maxHeap[0]:
          heappop(maxHeap)
          heappush(maxHeap, -nums[i])

    # the root of the heap has the Kth smallest number
    return -maxHeap[0]
```

- 手写heappush
  - Heaps are arrays for which heap[k] <= heap[2*k+1] and heap[k] <= heap[2*k+2] for all k

```python
# 其他: 1382. Balance a Binary Search Tree

def heappush(heap, item):
    # 将新元素添加到列表末尾
    heap.append(item)
    # 获取新元素的索引
    i = len(heap) - 1
    # 向上迭代调整堆
    while i > 0:
        parent = (i - 1) // 2
        # 如果新元素比其父节点小，则交换它们的位置
        if heap[i] < heap[parent]:
            heap[i], heap[parent] = heap[parent], heap[i]
            i = parent
        else:
            break

# 示例
heap = [3, 8, 10, 11, 9, 20]
print("原始堆:", heap)
heappush(heap, 5)
print("添加元素后的堆:", heap)
```
