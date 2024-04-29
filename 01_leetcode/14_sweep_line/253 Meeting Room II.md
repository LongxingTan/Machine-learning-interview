# 253 Meeting Room II
[https://leetcode.com/problems/meeting-rooms-ii/](https://leetcode.com/problems/meeting-rooms-ii/)


## solution

- heap
```python
# 每个会议入堆前都判断堆里是否有会议结束，如果有就让它出堆。

```
时间复杂度：O() <br>
空间复杂度：O()

- sort
```python
# 差分数组。使用一个计数器cnt表示当前正在召开的会议，然后从小到大遍历所有的时间点。若当前时间点有会议召开，那么就将cnt加上1，反之，若当前时间有会议结束，那么就将cnt减去1

```
时间复杂度：O() <br>
空间复杂度：O()

- 扫描线
