# ETA


- They represent the physical map as a graph

- They compute ETA by finding the shortest path in the directed weighted graph

- They don't use Dijkstra’s algorithm because it wouldn't scale with O(n logn) time complexity

- They partition the graph and then precompute the best path within each partition

- Partitioning graph reduces the time complexity from O(n^2) to O(n)

- They populate the edge weights of the graph with traffic information

- They do map matching to find accurate ETA

- They use the Kalman filter and Viterbi algorithm for map matching


## reference
- [https://newsletter.systemdesign.one/p/uber-eta](https://newsletter.systemdesign.one/p/uber-eta)
