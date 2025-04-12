# 数学类

尤其是一些矩阵运算与随机数的

## 随机

```python
import random

random.randint(start, stop)  # 都是闭区间

# random.choice 在list, tuple, string时间复杂度是O(1)；在set, dictionary时间复杂度是O(n), 集合和字典是无序、没有索引, 需要遍历来随机选择
random.choice(mylist)

# 打乱顺序
cards = list(range(53))
random.shuffle(cards)
```

Fisher–Yates shuffle

```text
-- To shuffle an array a of n elements (indices 0..n-1):
for i from n−1 down to 1 do
     j ← random integer such that 0 ≤ j ≤ i
     exchange a[j] and a[i]
```

```python
from random import randint

def randomize (arr, n):
    # Start from the last element and swap one by one. We don't need to run for the first element that's why i > 0
    for i in range(n - 1, 0, -1):
        j = randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr
```

## 位运算

```python
# & 按位与运算符：参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0
# | 按位或运算符：只要对应的二个二进位有一个为1时，结果位就为1
# ^ 按位异或运算符：当两对应的二进位相异时，结果为1; Exclusive OR 异或
# ~ 按位取反运算符：对数据的每个二进制位取反,即把1变为0,把0变为1。~x 类似于 -x-1
# << 左移动运算符：运算数的各二进位全部左移若干位，由"<<"右边的数指定移动的位数，高位丢弃，低位补0. 左移1位相当于乘2，x << n等价于x * (2 ^ n)
# >> 右移动运算符：把">>"左边的运算数的各二进位全部右移若干位，">>"右边的数指定移动的位数
# >>= 向右移位，然后赋值给左边变量。例如 a >>= 2, a向右移动两位，然后把结果赋值给a。
# <<= 向左移位，然后赋值给左边变量。例如 a <<= 2, a向左移动两位，然后把结果赋值给a。
# &= 按位与，然后赋值给左边变量。例如 a &= 2, a和2按位与，然后把结果赋值给a。
# |= 按位或，然后赋值给左边变量。例如 a |= 2, a和数字2按位或，然后把结果赋值给a。
# ^= 按位异或，然后赋值给左边变量。例如 a ^= 2, a和数字2按位异或，然后把结果赋值给a。
```

- separate rightest bit of 1， x & (-x)
- unset rightest bit of 1, x & (x-1)
- radix sort
- int 转化为 bit字符串 `bin(x)[2:]` 或 `format(n, '08b')`

| Operation    | Code                                 | Example                                       | Note                                          |
| ------------ | ------------------------------------ | --------------------------------------------- | --------------------------------------------- |
| union        | `a \| b`                             | `1010 \| 0110 == 1110`                        |                                               |
| difference   | `a & ~b`                             | `1010 & ~0110 == 1000`                        |                                               |
| intersection | `a & b`                              | `1010 & 0110 == 0010`                         |                                               |
| add          | `a                     \|= 1 << idx` | `1010                      \| 1 << 2 == 1110` | `idx` 2 is 0-indexed from the right e.g. 3210 |
| discard      | `a &= ~(1 << idx)`                   | `1010 & ~(1 << 3) == 0010`                    | `idx` 3 is 0-indexed from the right e.g. 3210 |
| contains?    | `bool(a & (1 << idx))`               | `1010 & (1 << 3) == True`                     | `idx` 3 is 0-indexed from the right e.g. 3210 |
