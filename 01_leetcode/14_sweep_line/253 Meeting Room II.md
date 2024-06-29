# 253 Meeting Room II
[https://leetcode.com/problems/meeting-rooms-ii/](https://leetcode.com/problems/meeting-rooms-ii/)


## solution

- heap
```python
# 每个会议入堆前都判断堆里是否有会议结束，如果有就让它出堆。

class Solution:
    def min_meeting_rooms(self, intervals: List[Interval]) -> int:
        if not intervals:
            return 0

        free_rooms = []
        intervals.sort(key= lambda x: x.start)
        heapq.heappush(free_rooms, intervals[0].end)

        for i in intervals[1:]:
            if free_rooms[0] <= i.start:
                heapq.heappop(free_rooms)

            # If a new room is to be assigned, then also we add to the heap,
            # If an old room is allocated, then also we have to add to the heap with updated end time.
            heapq.heappush(free_rooms, i.end)

        return len(free_rooms)
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
```python
class Solution:
    def minMeetingRooms(self, intervals):
        room=[]
        for i in intervals:
            room.append((i.start, 1))
            room.append((i.end, -1))

        tmp=0
        ans=0
        room=sorted(room)
        for idx, cost in room:
            tmp += cost
            ans = max(ans, tmp)
        return ans
```

- 前缀和
```python
class Solution:
    def minMeetingRooms(self, intervals):
        room = {}
        # 开始时间+1，结束时间-1
        for i in intervals:
            room[i.start] = room.get(i.start, 0) + 1
            room[i.end] = room.get(i.end, 0) - 1

        ans = 0
        tmp = 0
        for i in sorted(room.keys()):
            tmp = tmp + room[i]
            ans = max(ans, tmp)
        return ans
```


## follow up

[*252. Meeting Rooms](../10_greedy/56.%20Merge%20Intervals.md)
