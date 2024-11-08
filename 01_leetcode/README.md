# Leetcode
> leetcode在面试时常被认为是衡量CS水平的重要指标，因此很多公司需要好好刷题，精刷大于泛刷，熟练度大于题量


## 1. offline 刷题策略
一般大厂的面试非常规范和模版化，严格按照标准评分，学会踩得分点和传达signal。想到某个方向可以加多少分，写出一个实现可以加多少分，有bug扣多少分，答错复杂度扣多少分等，因此需要尽量提高熟练度。

- 1、初步掌握常见数据结构和算法，包括一些类型的解题模版
- 2、按tag刷题，模版的基础上，不断强化练习。尝试多种解法实现
- 3、按热度刷题，参加周赛，刷题量达到满意要求。结合问题交流、复杂度分析、头脑测试
- 4、面试前，针对面试公司tag和最新面经进行重点练习。练习时边说边写


## 2. online 面试解题过程
除了快速给出bug free的代码，面试过程中的沟通交流也非常重要。做题前的寒暄适当精炼，个人和项目介绍不必随意展开，给做题留足时间。

- (1) 面试官出题
- (2) 面试者读题并提问 **clarification questions**
  - 有面试官会故意遗漏内容，不明白的地方一定问清楚。可以通过列举test case，确保真正理解问题并涵盖所有情况。
  - 需求确定好之后，确认输入输出的类型和边界，input/output type, 特殊情况找没找到的时候返回什么
  - input element的范围
    - 数字： 全是正数或者全是负数 或者返回必须正数或负数
    - size 以及edge cases of null or empty
    - string：是不是都是English letters (A-Z)，有没有特殊符号，有的话，会有哪些出现
    - input element 有无重复
    - standard input, empty input, malformed input, long input, null input, other edge input
  - 能不能改input (1D/2D array比较常见）
  - 时间空间复杂度要求
    - 需不需要in place
    - 有没有内存限制
  - 区分subarrays和subsequence等
  - 输入是否有重复样本，输出是否允许重复
- (3) 面试官回答问题保证面试者正确理解题目
- (4) 面试者思考怎么做，有想法之后先和面试官讲明白思路，并得到认同
  - 从逻辑/模拟角度思考如何做这个任务(第一性原则出发)。没有思路时，想想更简单、数量更少时如何处理. 
  - Whenever coming across a solution, talk it out and discuss it with the interviewer
  - 2分钟内还没有思路，主动和面试官要提示(hint)
  - 可以询问：Am I on the right direction?
  - 想不到最优的方法也可以先跑一个可行解再优化，不确定是否最有可询问："Could we have a better solution?"  
  - 熟悉的题目也可以先说一下穷举/暴力做法，然后简单分析下重复操作或者引入高效的数据结构，从而引入到最优解
  - 没有得到面试官认同你的算法之前不要着急动手写代码
- (5) 面试者写代码实现算法，一定要确认好思路后才开始写
  - 用什么数据结构，什么算法。讲思路的时候，一定说清楚为什么选择这个数据结构，结合有代表性的test case说明
  - During writing the code, whenever coming across edge cases, discuss with the interviewer about how to handle the edge case
- (6) 跑测试实例验证代码正确
  - I just finished my coding, now I need to run several test cases to see if I covered all the edge cases and if there is any bugs I missed
  - 注意可能需要自己写tests，给出corner case test
  - **dry run**
    - 选择能够覆盖各种边界情况和不同输入的例子
    - 列出你算法的所有变量的初始状态
    - 逐步执行算法，逐步更新这些变量的值
    - 每个关键点，检查你的变量是否符合预期
  - 空，一个数，两个一样的数(重复)
- (7) 面试者讲述自己算法的时间、空间复杂度
- (8) 面试官确认没问题后准备follow up题目或者下一道题目
  - 算法相关技术也可能被问到
    - rate limiter
    - 不同的cache机制
    - geo data (quad tree)
    - 文本搜索提示 (trie)


## 3. 公司特点

[OA真题](https://github.com/perixtar/2024-Tech-OA)

**Google**
- LeetCode: 不按tag非原题, hard多(DP, DFS, Backtracking, Trie)，极其看重解决问题的思考过程，自己写test case。面经题仍然有助于准备

**Meta**
- LeetCode: tag高频，35分钟两道题, 看重结果bug free和最优解
- ML system design: recommendation, video search, harmful content detection

**Amazon**
- LeetCode: Grind75, 20分钟一道
- BQ: 每一轮BQ占30分钟+, 2个LP

**Tiktok**
- LeetCode: 不按tag
- ML: 可能有国内面试习惯，深挖简历项目，讲清整个领域来龙去脉


## 4. 基础

**精读**
- [https://blog.faangshui.com/p/how-to-talk-to-the-interviewer](https://blog.faangshui.com/p/how-to-talk-to-the-interviewer)


### Python
- 垃圾回收机制gc
- 全局锁GIL
- python里的list和dict是怎么实现的
- python里的with怎么实现的
- [异步与协程](https://zhuanlan.zhihu.com/p/25228075)


### CS基础
- [Introduction to Computer Science - Harvard CS50x](https://cs50.harvard.edu/x/)
- [Structure and Interpretation of Computer Programs - UC Berkeley CS 61A](https://cs61a.org/)
- [How to Design Programs](https://book.douban.com/subject/30175977/)
- [深入理解计算机系统 - CSAPP](https://book.douban.com/subject/5333562/)
- [The Art of Computer Programming - TAOCP](https://www-cs-faculty.stanford.edu/~knuth/taocp.html)
- [代码大全](https://book.douban.com/subject/1477390/)
- [UNIX 编程艺术](https://book.douban.com/subject/11609943/)
- [重构：改善既有代码的设计](https://book.douban.com/subject/4262627/)


### 数据结构和算法
- [数据结构- 学堂在线 - 邓俊辉](https://next.xuetangx.com/course/THU08091000384/)
- [算法 - Stanford](https://www.coursera.org/specializations/algorithms)
- [Algorithms](https://book.douban.com/subject/1996256/)
- [算法导论 - CLRS](https://book.douban.com/subject/20432061/)


## 可参考刷题列表
- [代码随想录](https://programmercarl.com/)
- [LeetCode 101：和你一起你轻松刷题](https://github.com/changgyhub/leetcode_101/)
- [blind75](https://leetcode.com/list/xi4ci4ig/)
- [Neetcode300](https://neetcode.io/practice)
- [Leetcode面试高频题分类刷题总结](https://zhuanlan.zhihu.com/p/349940945)
- [花花酱 LeetCode Problem List 题目列表](https://zxi.mytechroad.com/blog/leetcode-problem-categories/)
- [时间复杂度-Abdul Bari's Algorithm Playlist](https://www.youtube.com/playlist?list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O)
- [Data Structures Easy to Advanced Course - Full Tutorial from a Google Engineer and ACM ICPC World Finalist](https://www.youtube.com/playlist?list=PLDV1Zeh2NRsB6SWUrDFW2RmDotAfPbeHu)
- [灵茶山艾府](https://github.com/EndlessCheng)

**online assessment**
- [https://www.glassdoor.com/index.htm](https://www.glassdoor.com/index.htm)
- [https://www.careercup.com/](https://www.careercup.com/)
