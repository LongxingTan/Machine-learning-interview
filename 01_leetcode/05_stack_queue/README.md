# 栈与队列

- pop前注意检查是不是为空
 

## 栈
[Design a stack in python](https://www.geeksforgeeks.org/design-a-stack-that-supports-getmin-in-o1-time-and-o1-extra-space/)

[stack in java](https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/Stack.java.html)

栈的一些题目思路并不容易，比如计算器224/227，decoder string 394


## 单调栈 monotone stack
单调栈通常是一维数组，用于解决数组中找出每个数字左右/右边第一个大于／小于该数字的位置或者数字；单调的意思是保留在栈或者队列中的数字是单调递增或者单调递减的

- 直观感受的话，每次遇到比之前大的元素才更新之前的，否则暂时存入栈
- 比如找右边比自己大的元素，如果右边元素太小，该元素先在栈里等待直到遇到大的；遇到了同时更新结果. 遇到了就直接出栈，因此得到的是第一个比自己大的
- 单调栈的维护是 O(n) 级的时间复杂度，因为所有元素只会进入栈一次，并且出栈后再也不会进栈了


## 队列
[Introduction to Circular Queue](https://www.geeksforgeeks.org/introduction-to-circular-queue/)
