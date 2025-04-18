# LeetCode

> LeetCode在面试时常被认为是衡量CS水平的重要指标。**精刷大于泛刷，熟练度大于题量**。因此，尽量提高熟练度，并踩对得分点和signal

## 1. offline 刷题策略

大厂面试相对规范和模版化，按四个维度评分(communication, problem solving, coding, verification)。想到某个方向可以加多少分，写出一个实现可以加多少分，有bug扣多少分，答错复杂度扣多少分等。

- 1、如果零基础，先掌握常见的数据结构和算法，包括常见类型的解题模版
- 2、按tag刷题，熟悉模版的基础上，不断强化练习，并尝试多种解法实现
- 3、按热度刷题，参加周赛，刷题量达到满意要求。练习边说边写，结合题目交流、复杂度分析、头脑测试
- 4、面试前，针对面试公司tag和最新面经重点练习

## 2. online 面试解题过程

面试过程中，重要的是: 快速给出bug free代码，有良好的**沟通**和**思路介绍**。做题前的寒暄、项目介绍不必随意展开(1~2min)，留够做题时间。

- (1) 面试官出题
- (2) 面试者读题并 **clarification question**
  - 不明白的地方一定问清楚，而不是随意假设或略过，有面试官会故意遗漏内容。可以通过列举test case，来确保理解问题并涵盖所有情况。
  - 功能需求确定好之后，确认输入/输出的类型和边界(input/output type)
    - 输入数字全是正数或者全是负数， null or empty
    - 输入string是不是都是English letters (A-Z)，有没有特殊符号
    - 输入是否有重复样本，输出是否允许重复
    - standard input, empty input, malformed input, long input, null input, other edge input
    - 能不能改input，immutable or not (1D/2D array比较常见）
    - 特殊情况输出返回什么
  - 确认时间空间复杂度要求
    - 要不要in place
    - 有没有内存限制
- (3) 面试官回答问题保证面试者正确理解题目
- (4) 面试者思考怎么做，先和面试官讲明白思路，并得到认同。思路未得到认同前不要急于写代码
  - Whenever coming across a solution, talk it out and discuss it with the interviewer
  - 第一性原则出发，从逻辑/模拟角度思考如何解决问题。没有思路时，想想更简单、数量更少时
  - 介绍思路时，可以询问: Am I in the right direction?
  - 2分钟内还没有思路，可以主动和面试官要提示(hint)  
  - 想不到最优的方法也可以先跑一个可行解再优化，不确定是否最优可询问: Could we have a better solution?
  - 熟悉的题目也可以先说一下穷举/暴力做法，通过分析重复操作或引入高效的数据结构，从而引入最优解
- (5) 确认好思路后写代码实现算法
  - 用什么数据结构，什么算法。讲思路的时候，说清楚为什么选择这个数据结构
  - During writing the code, whenever coming across edge cases, discuss with the interviewer about how to handle the edge case
- (6) 测试验证代码正确性
  - I just finished my coding, now I need to run several test cases to see if I covered all the edge cases and if there is any bugs I missed
  - 注意可能需要自己写tests，给出corner case test
  - **dry run**
    - 选择能够覆盖各种边界情况和不同输入的例子(test cases)
    - 列出你算法的所有变量的初始状态
    - 逐步执行算法，逐步更新这些变量的值
    - 每个关键点，检查你的变量是否符合预期
  - 空，一个数，两个一样的数(重复)
- (7) 讲述自己算法的时间、空间复杂度
- (8) 面试官确认没问题后准备follow up题目或者下一道题目

## 3. 公司特点

> - [OA真题](https://github.com/perixtar/2024-Tech-OA)
> - [大厂的面试的差异和重点](https://www.1point3acres.com/bbs/thread-1021931-1-1.html)

**Google**

- LeetCode: 非tag原题多，hard多(DP, DFS, Backtracking, Trie)，极其看重解决问题的思考过程，自己写test case。面经题仍有帮助
- [Googleyness & Leadership Interview Questions](https://igotanoffer.com/blogs/tech/googleyness-leadership-interview-questions#googleyness)

**Meta**

- LeetCode: tag原题，35分钟两道题, 看重bug free和最优解
- ML system design: recommendation, video search, harmful content detection

**Microsoft**

- LeetCode: 基础算法、数据结构的快速实现

**Amazon**

- LeetCode: Grind75， 15分钟一道题，原题，Graph
- BQ: 每一轮BQ占30分钟+， 2个LP

**Tiktok**

- LeetCode: 不按tag
- ML: 可能有国内面试习惯，深挖简历项目，MLE讲清领域来龙去脉

## 4. 刷题

### 4.1 参考刷题列表

> 重点: 刷透高频题，不要假设哪种类别肯定不会考

- [blind75](https://leetcode.com/list/xi4ci4ig/)
- [代码随想录](https://programmercarl.com/)
- [LeetCode 101：和你一起你轻松刷题](https://github.com/changgyhub/leetcode_101/)
- [Neetcode300](https://neetcode.io/practice)
- [灵茶山艾府](https://github.com/EndlessCheng)
- [Leetcode面试高频题分类刷题总结](https://zhuanlan.zhihu.com/p/349940945)
- [花花酱 LeetCode Problem List 题目列表](https://zxi.mytechroad.com/blog/leetcode-problem-categories/)
- [时间复杂度-Abdul Bari's Algorithm Playlist](https://www.youtube.com/playlist?list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O)
- [Data Structures Easy to Advanced Course - Full Tutorial from a Google Engineer and ACM ICPC World Finalist](https://www.youtube.com/playlist?list=PLDV1Zeh2NRsB6SWUrDFW2RmDotAfPbeHu)

### 4.2 Python

- 垃圾回收机制gc
- 全局锁GIL
  - 因为引用计数所以需要全局锁，因为全局锁所以影响多线程
- python里的list和dict是怎么实现的
- python里的with怎么实现的
- [异步与协程](https://zhuanlan.zhihu.com/p/25228075)
  - [什么是协程？](https://zhuanlan.zhihu.com/p/172471249)
  - [也来谈谈协程 - invalid s的文章 - 知乎](https://zhuanlan.zhihu.com/p/147608872)
  - [进程与线程](https://zhuanlan.zhihu.com/p/46368084)
- 数据结构
  - mutable(可变): list, dict, set
  - immutable(不可变): int, float, str, tuple

### 4.3 Java

- [java guide](https://javaguide.cn/home.html)

### 4.4 CPP

- [Containers library](https://en.cppreference.com/w/cpp/container)

### 4.5 CS基础与DSA

- [Introduction to Computer Science - Harvard CS50x](https://cs50.harvard.edu/x/)
- [Structure and Interpretation of Computer Programs - UC Berkeley CS 61A](https://cs61a.org/)
- [How to Design Programs](https://book.douban.com/subject/30175977/)
- [深入理解计算机系统 - CSAPP](https://book.douban.com/subject/5333562/)
- [The Art of Computer Programming - TAOCP](https://www-cs-faculty.stanford.edu/~knuth/taocp.html)
- [代码大全](https://book.douban.com/subject/1477390/)
- [UNIX 编程艺术](https://book.douban.com/subject/11609943/)
- [重构：改善既有代码的设计](https://book.douban.com/subject/4262627/)
- [算法 - Stanford](https://www.coursera.org/specializations/algorithms)
- [Algorithms](https://book.douban.com/subject/1996256/)
- [算法导论 - CLRS](https://book.douban.com/subject/20432061/)
- MIT 6.824 分布式系统
- MIT 6.s081 操作系统
- CMU 15445 数据库
