# 前缀和

prefix有多种类型，如prefix tree, prefix hashmap

使用双指针解决子数组只和的前提是：正数数组。当数组中有负数是，无法确定一个固定的方向，此时需要的就是前缀和。

- 计算从下标0到各个位置的数组cumsum，把子数组看作是之前数组之和新加一个数字
- prefix sum - target有无出现和出现的次数
