# 207 Course Schedule
[https://leetcode.com/problems/course-schedule/](https://leetcode.com/problems/course-schedule/)


## solution

- 建图，判断图是否有环
- 标准的拓扑排序写法

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = collections.defaultdict(list)
        degree = [0] * numCourses  # 也可以使用collections.defaultdict(int)

        for (cur, pre) in prerequisites:
            graph[pre].append(cur)  # bfs往后走所以记录后面
            degree[cur] += 1  # 后面是否可开始依赖前面

        start_course = [i for (i, d) in enumerate(degree) if d == 0]
        queue = collections.deque(start_course)
        visited = 0
        while queue:
            cur = queue.popleft()
            visited += 1
            for adj in graph[cur]:  # graph记录后续可以开始的课程
                degree[adj] -= 1  # 后续课程的前依赖 - 1
                if not degree[adj]:
                    queue.append(adj)
        return visited == numCourses
```


- 建立队列存储不需要先行课程的课程，从队列中取出元素，找到依赖该元素的后续课程。如果后续课程不再依赖其他课程，则加入队列

```python
import collections

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        pres = collections.defaultdict(set)
        courses = collections.defaultdict(set)

        for x, y in prerequisites:
            pres[x].add(y)
            courses[y].add(x)

        # 注意是从全体课程中选择可开始
        no_pre_stack = [i for i in range(numCourses) if not pres[i]]
        count = 0
        while no_pre_stack:
            take_course = no_pre_stack.pop()
            count += 1

            for x in courses[take_course]:
                pres[x].remove(take_course)
                if not pres[x]:
                    no_pre_stack.append(x)

        return count == numCourses
```
时间复杂度：O(E+V) <br>
空间复杂度：O(E+V)


## Follow-up

### 调度类

[210. Course Schedule II](./210.%20Course%20Schedule%20II.md)

[630 Course Schedule III](https://leetcode.com/problems/course-schedule-iii/description/)
```python
class Solution:
    def schedule_course(self, courses: List[List[int]]) -> int:  
        # https://algo.monster/liteproblems/630
        courses.sort(key=lambda c: c[1])  # 按需求日期排序

        heap = list()        
        total = 0  # 优先队列中所有课程的总时间

        for ti, di in courses:
            if total + ti <= di:
                total += ti
                heapq.heappush(heap, -ti)  # 如果不足，把大的pop处理，因此用负树
            elif heap and -heap[0] > ti:
                total -= -heap[0] - ti
                heapq.heappop(heap)
                heapq.heappush(heap, -ti)

        return len(heap)
```

[1462 Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/description/)
```python
class Solution:
    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        graph = collections.defaultdict(list)
        degree = [0] * numCourses
        pre_lookup = collections.defaultdict(set)

        for pre, cur in prerequisites:
            graph[pre].append(cur)
            degree[cur] += 1

        queue = collections.deque([i for i in range(numCourses) if degree[i] == 0])      
        while queue:
            node = queue.popleft()
            for cur in graph[node]:
                pre_lookup[cur].add(node)
                pre_lookup[cur].update(pre_lookup[node])

                degree[cur] -= 1

                if degree[cur] == 0:
                    queue.append(cur)

        res = []
        for q in queries:
            if q[0] in pre_lookup[q[1]]:
                res.append(True)
            else:
                res.append(False)
        return res
```

[1834 Single-Threaded CPU](https://leetcode.com/problems/single-threaded-cpu/description/)

- dijkstra: 用heap
```python
class Solution:
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        tasks = sorted((earliest_time, processing_time, i) for i, (earliest_time, processing_time) in enumerate(tasks))

        res = []
        heap = []
        time = tasks[0][0]

        for earliest_time, processing_time, i in tasks:
            while heap and time < earliest_time:
                processing, idx, earliest = heapq.heappop(heap)
                res.append(idx)
                time = max(time, earliest) + processing
            heapq.heappush(heap, (processing_time, i, earliest_time))

        while heap:
            res.append(heapq.heappop(heap)[1])
        return res
```

- 超时: 用heap 代替BFS的deque
```python
class Solution:
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        new_tasks = []
        for i, task in enumerate(tasks):
            new_tasks.append([task[1], i, task[0]])

        new_tasks.sort(key=lambda x: (x[2], x[0]))

        res = []
        global_time = new_tasks[0][2]

        pq = []
        heapq.heappush(pq, new_tasks[0])

        while pq:
            processing_time, no, earliest_time = heapq.heappop(pq)
            res.append(no)

            global_time += processing_time

            for task in new_tasks:
                if task[1] not in res and global_time >= task[2] and task not in pq:
                    heapq.heappush(pq, task)
        return res
```

[621 Task Scheduler](../10_greedy/621.%20Task%20Scheduler.md)


[2365 Task Scheduler II](https://leetcode.com/problems/task-scheduler-ii/description/)
```python
class Solution:
    def taskSchedulerII(self, tasks: List[int], space: int) -> int:
        my_dict = collections.defaultdict(int)
        time = 0
        for task in tasks:
            time = max(time + 1, my_dict[task])
            my_dict[task] = time + space + 1
        return time
```


[1029 Two City Scheduling](https://leetcode.com/problems/two-city-scheduling/description/)
```python
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        # 主要是差值，其次是绝对值
        cost_diff = [(i - j, i, t) for t, (i, j) in enumerate(costs)]
        cost_diff.sort()

        candidate_t = [i[2] for i in cost_diff]
        res = 0
        for i in candidate_t[: len(costs)//2]:
            res += costs[i][0]
        
        for i in candidate_t[len(costs)//2:]:
            res += costs[i][1]
        return res
```

[*1229 Meeting Scheduler](https://leetcode.com/problems/meeting-scheduler/description/)
- 双指针
```python
class Solution:
    def earliest_appropriate_duration(self, slots1, slots2, duration):
        index1 = 0
        index2 = 0
        while index1 < len(slots1) and index2 < len(slots2):
            left1, right1 = slots1[index1].start, slots1[index1].end
            left2, right2 = slots2[index2].start, slots2[index2].end
            if (min(right1, right2) - max(left1, left2)) >= duration:
                return Interval(max(left1, left2), max(left1, left2)+duration)
            if right1 < right2:
                index1 += 1
            else:
                index2 += 1
        return Interval(-1, -1)
```

[1335 Minimum Difficulty of a Job Schedule](../09_dynamic_program/1335%20Minimum%20Difficulty%20of%20a%20Job%20Schedule.md)
```python

```

[1235 Maximum Profit in Job Scheduling](../09_dynamic_program/1235%20Maximum%20Profit%20in%20Job%20Scheduling.md)


[2050. Parallel Courses III](https://leetcode.com/problems/parallel-courses-iii/description/)
```python
class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:

        graph = collections.defaultdict(list)
        degree = [0] * (n + 1)
        complete_time = collections.defaultdict(list)
        res_time = [0] * (n + 1)

        # build graph
        for pre, cur in relations:
            graph[pre].append(cur)
            degree[cur] += 1

        queue = collections.deque([i for i in range(1, n + 1) if degree[i] == 0])

        for node in queue:
            res_time[node] = time[node-1]

        while queue:
            node = queue.popleft()
            for nex in graph[node]:
                degree[nex] -= 1
                complete_time[nex].append(res_time[node])
                if degree[nex] == 0:
                    queue.append(nex)
                    res_time[nex] = max(complete_time[nex]) + time[nex-1]
        return max(res_time)
```

[*2402. Meeting Rooms III](https://leetcode.com/problems/meeting-rooms-iii/description/)
```python
# 3 heap: https://zhuanlan.zhihu.com/p/567648312

```

[1882. Process Tasks Using Servers](https://leetcode.com/problems/process-tasks-using-servers/description/)
- two heap 思路比较巧妙: 一个存储正在工作的server, 一个存储空闲的server
```python
# 1. 更新时间, 每个单位时间更新一个任务，进来一个新任务，意味着至少到了当前时间
# 2. 如果当前空闲服务器堆为空，等待最早完成的服务器完成工作, 相当于当前时间往前空转
# 3. now时刻，有多少服务器完成了工作，弹出加入到空闲堆
# 4. 空闲堆中选择要求的服务器作为本工作完成的服务器

class Solution:
    def assignTasks(self, servers: List[int], tasks: List[int]) -> List[int]:
        working_servers = []  # (空闲时间，index)
        idling_servers = []  # (权重，index)

        for i, server in enumerate(servers):
            heapq.heappush(idling_servers, [server, i])

        time = 0
        res = []
        for index, task in enumerate(tasks):  # 每个单位时间一个task abailable
            if time < index:
                time = index

            if not idling_servers:
                time = working_servers[0][0]

            while working_servers and working_servers[0][0] == time:
                _, i, server = heapq.heappop(working_servers)
                heapq.heappush(idling_servers, [server, i])

            server, i = heapq.heappop(idling_servers)
            heapq.heappush(working_servers, [time + task, i, server])
            res.append(i)
        return res
```


- straight [需要debug, 没有全过]
```python
class Solution:
    def assignTasks(self, servers: List[int], tasks: List[int]) -> List[int]:
        time = 0
        servers_available_time = [0] * len(servers)

        res = []
        stack = [tasks[0]]

        while stack:
            if min(servers_available_time) <= time:
                processing_time = stack.pop(0)
                candidates = [i for i, x in enumerate(servers_available_time) if x <= time]

                candidates_servers = [servers[i] for i in candidates]
                i = candidates_servers.index(min(candidates_servers))
                res.append(candidates[i])
                servers_available_time[candidates[i]] = time + processing_time

            time += 1
            if time < len(tasks):
                stack.append(tasks[time])

        return res
```

[1386. Cinema Seat Allocation](https://leetcode.com/problems/cinema-seat-allocation/description/)
```python

```

Task scheduler
- multiple ec2 machine 处理 task，每个machine 能同时处理10个jobs，then we want to have an algorithm that when can we start for the next job
```python
# multiple heap
```


### 拓扑排序类

[*261. Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)

```python
# 图联通且无环的必要条件: 1.edges数目必须等于n-1 2.连通图总集合数为1
import collections

class Solution:
    def validTree(self, n, edges):
        if n == 0:
            return False
        if len(edges) != n - 1:
            return False
        graph = collections.defaultdict(list)
        for e in edges:
            graph[e[0]].append(e[1])
            graph[e[1]].append(e[0])

        q = [0]
        visited = set([0])
        while q:
            node = q.pop(0)
            for i in graph[node]:
                if i in visited:
                    continue
                q.append(i)
                visited.add(i)
        return len(visited) == n
```

```python
# 显式检查是否成环
class Solution:
    def validTree(self, n, edges):
        if n == 0:
            return False
        if len(edges) != n - 1:
            return False
        graph = collections.defaultdict(list)
        for e in edges:
            graph[e[0]].append(e[1])
            graph[e[1]].append(e[0])

        q = [(0, -1)]  # (current_node, parent_node)
        visited = set([0])
        while q:
            node, parent = q.pop(0)
            for i in graph[node]:
                if i == parent:
                    continue  # Skip the parent node
                if i in visited:
                    return False  # Found a cycle
                q.append((i, node))
                visited.add(i)
        return len(visited) == n
```

- union find
```python

```

[*269 Alien Dictionary](./269%20Alien%20Dictionary.md)
