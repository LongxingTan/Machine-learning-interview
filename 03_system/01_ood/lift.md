# 电梯设计

## Low level Design Diagram

```plantuml
actor client
node app
database db
db -> app
app -> client
```

## 代码

```python
# https://leetcode.com/discuss/interview-question/1612174/OOD-Elevator
from enum import Enum
from heapq import heappush, heappop

class Direction(Enum):
    up = 'UP'
    down = 'DOWN'
    idle = 'IDLE'


class RequestType(Enum):
    external = 'EXTERNAL'
    internal = 'INTERNAL'


class Request:
    def __init__(self, origin, target, typeR, direction):
        self.target = target
        self.typeRequest = typeR
        self.origin = origin
        self.direction = direction


class Button:
    def __init__(self, floor):
        self.floor = floor


class Elevator:
    def __init__(self, currentFloor):
        self.direction = Direction.idle
        self.currentFloor = currentFloor
        self.upStops = []
        self.downStops = []

    def sendUpRequest(self, upRequest):
        if upRequest.typeRequest == RequestType.external:
            heappush(self.upStops, (upRequest.origin, upRequest.origin))

        heappush(self.upStops, (upRequest.target, upRequest.origin))

    def sendDownRequest(self, downRequest):
        if downRequest.typeRequest == RequestType.external:
            heappush(self.downStops, (-downRequest.origin, downRequest.origin))
        heappush(self.downStops, (-downRequest.target, downRequest.origin))

    def run(self):
        while self.upStops or self.downStops:
            self.processRequests()

    def processRequests(self):
        if self.direction in [Direction.up, Direction.idle]:
            self.processUpRequests()
            self.processDownRequests()
        else:
            self.processDownRequests()
            self.processUpRequests()

    def processUpRequests(self):
        while self.upStops:
            target, origin = heappop(self.upStops)

            self.currentFloor = target

            if target == origin:
                print("Stopping at floor {} to pick up people".format(target))
            else:
                print("stopping at floor {} to let people out".format(target))

        if self.downStops:
            self.direction = Direction.down
        else:
            self.direction = Direction.idle

    def processDownRequests(self):
        while self.downStops:
            target, origin = heappop(self.downStops)

            self.currentFloor = target

            if abs(target) == origin:
                print("Stopping at floor {} to pick up people".format(abs(target)))
            else:
                print("stopping at floor {} to let people out".format(abs(target)))

        if self.upStops:
            self.direction = Direction.up
        else:
            self.direction = Direction.idle


elevator = Elevator(0)
upRequest1 = Request(elevator.currentFloor, 5, RequestType.internal, Direction.up)
upRequest2 = Request(elevator.currentFloor, 3, RequestType.internal, Direction.up)

downRequest1 = Request(elevator.currentFloor, 1, RequestType.internal, Direction.down)
downRequest2 = Request(elevator.currentFloor, 2, RequestType.internal, Direction.down)

upRequest3 = Request(4, 8, RequestType.external, Direction.up)
downRequest3 = Request(6, 3, RequestType.external, Direction.down)

elevator.sendUpRequest(upRequest1)
elevator.sendUpRequest(upRequest2)

elevator.sendDownRequest(downRequest1)
elevator.sendDownRequest(downRequest2)

elevator.sendUpRequest(upRequest3)
elevator.sendDownRequest(downRequest3)

elevator.run()
```
