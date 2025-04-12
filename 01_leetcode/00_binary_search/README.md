# 二分查找

- 二分查找将线性时间提升到了对数时间，大大缩短了搜索时间，时间复杂度为 O(log n)
- Java或C，start+end可能overflow，需要用start + (end - start) // 2

## 总结

**确认点**

- 有无重复值
- 是否排序或是否单调
  - 类似双指针的原理，必须能够明确，然后左右只能移动一个
  - 双指针类型的题， 指针通常是一步一步移动的，而在二分查找里，指针每次移动半个区间长度
- 明确范围，target可能在左右边界之外

**注意点**

- 明确开闭区间，每一次边界处理都根据区间的定义来操作. 同时参考bisect中的含义（输出比目标值大的最左或最右）
  - 左闭右闭 [left, right]
  - 左闭右开 half-closed [left, right)
- 1.初始化右边界 right设置为len(nums)还是len(nums-1)
- 2.while条件，即何时退出循环，避免无限循环
  - 如果最后区间只剩下一个数或者两个数，自己的写法是否会陷入死循环，如果某种写法无法跳出死循环，则考虑尝试另一种写法
  - 避免死循环用 (start + 1) < end，注意左右迭代时是否加1
- 3.左右边界更新，如何收缩检索区间
- 4.输出, 返回 left，right，或 right - 1

**分类**

- 查找和目标值完全相等的数
- 查找第一个不小于目标值的数，变形为查找最后一个小于目标值的数
- 查找第一个大于目标值的数，变形为查找最后一个不大于目标值的数
- 用子函数当作判断关系（通常由 mid 计算得出）
- 其他（通常 target 值不固定）
  - peak
- 2D查找

```python
def binary_search(key, nums):
    left = 0
    right = len(nums)-1

    while left <= right:
        mid = (left + right) // 2

        if key < nums[mid]:
            right = mid - 1
        elif key > nums[mid]:
            left = mid + 1
        else:
            return mid
    return False

if __name__ == '__main__':
    print(binary_search(5, [1, 2, 4, 5, 9]))
```

## Reference

- [16-line-Python-solution-symmetric-and-clean-binary-search](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/discuss/14714/16-line-Python-solution-symmetric-and-clean-binary-search-52ms)
- [聊聊一看就会一写就跪的二分查找](https://zhuanlan.zhihu.com/p/343138037)
- [LeetCode Binary Search Summary 二分搜索法小结](https://www.cnblogs.com/grandyang/p/6854825.html)
