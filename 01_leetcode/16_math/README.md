# 数学类

尤其是一些矩阵运算与随机数的


## 随机

```python
import random

random.randint(start, stop)  # 都是闭区间

# random.choice 在list, tuple, string时间复杂度是O(1)；在set, dictionary时间复杂度是O(n), 集合和字典是无序、没有索引, 需要遍历来随机选择
random.choice(mylist)
```
