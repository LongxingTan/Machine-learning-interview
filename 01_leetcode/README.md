# Leetcode
leetcode经常被认为是衡量CS水平的最重要指标，因此要好好刷题，精刷大于泛刷，熟练度大于题量


## 刷题策略
规范大厂的面试非常模版化，严格按照标准评分，学会踩得分点。想到这个方向可以加多少分，写出一个实现可以加多少分，有bug扣多少分，答错复杂度扣多少分等，因此需要尽量提高熟练度。

- 1、初步掌握考察的常见数据结构和算法，尤其是常见类型的模版
- 2、按tag刷题，模版的基础上，不断强化练习。多种解法实现
- 3、按热度刷题，参加周赛，刷题量达到满意要求。结合问题交流、复杂度分析、头脑测试，
- 4、面试前，针对面试公司的tag和最新面经进行重点练习。自己练习时边说边写


## 面试解题过程
除了快速给出bug free的代码，过程中的沟通交流也非常重要。做题前需要的个人和简单项目介绍适当精炼，不要随意展开，给做题留足时间。

- (1) 面试官出题
- (2) 面试者读题并提问clarification questions
  - 有面试官故意遗漏内容，不明白的地方一定问清楚。也可以通过写test case，确保理解问题并涵盖所有情况。
  - 需求确定好之后，确认输入输出的类型和边界，input/output type, 特殊情况找没找到的时候返回什么
  - input element的范围
    - 数字： 全是正数或者全是负数 或者返回必须正数或负数
    - size 以及edge cases of null or empty
    - string：是不是都是English letters (A-Z)，有没有特殊符号，有的话，会有哪些会出现
    - input element 有无重复
    - standard input, empty input, malformed input, long input, null input, other edge input
  - 能不能改input (1D/2D array比较常见）
  - 时间空间复杂度要求
    - 需不需要in place
    - 有没有内存限制
  - 区分subarrays和subsequence等
  - 输入是否有重复样本，输出是否允许重复
- (3) 面试官回答问题来保证面试者正确理解题目
- (4) 面试者思考怎么做，有想法之后和面试官讲明白思路并得到认同
  - 从逻辑/模拟角度思考如何做这个任务(第一性原则出发)。没有思路时，想想更简单、数量更少时如何处理
  - 2分钟内还没有思路，主动和面试官要提示(hint)
  - 可以询问：Am I on the right direction?
  - 想不到最优的方法也可以先跑一个可行解再优化
  - 没有得到面试官认同你的算法之前不要着急动手写代码
- (5) 面试者写代码实现算法
  - 用什么数据结构，什么算法。讲思路的时候，一定说清楚为什么选择这个数据结构，结合有代表性的test case说明
- (6) 面试者把代码的大体结构讲一遍给面试官听
- (7) 跑test case来验证代码是否正确
  - 注意可能需要自己写tests
  - **dry run**
    - 选择能够覆盖各种边界情况和不同输入的例子
    - 列出你算法的所有变量的初始状态
    - 逐步执行算法，逐步更新这些变量的值
    - 每个关键点，检查你的变量是否符合预期
  - 空，一个数，两个一样的数(重复)
- (8) 面试者讲述自己写的算法的时间、空间复杂度
- (9) 面试官确认没问题后准备follow up题目或者下一道题目
  - 算法相关技术也可能被问到
    - rate limiter
    - 不同的cache机制
    - geo data (quad tree)
    - 文本搜索提示 (trie)


## 刷题列表
- [代码随想录](https://programmercarl.com/)
- [LeetCode 101：和你一起你轻松刷题](https://github.com/changgyhub/leetcode_101/)
- [blind75](https://leetcode.com/list/xi4ci4ig/)
- [Neetcode300](https://neetcode.io/practice)
- [Leetcode面试高频题分类刷题总结](https://zhuanlan.zhihu.com/p/349940945)
- [花花酱 LeetCode Problem List 题目列表](https://zxi.mytechroad.com/blog/leetcode-problem-categories/)
- [时间复杂度-Abdul Bari's Algorithm Playlist](https://www.youtube.com/playlist?list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O)
- [Data Structures Easy to Advanced Course - Full Tutorial from a Google Engineer and ACM ICPC World Finalist](https://www.youtube.com/playlist?list=PLDV1Zeh2NRsB6SWUrDFW2RmDotAfPbeHu)
- 本书目录中带*的为VIP题目


**online assessment**
- [https://www.glassdoor.com/index.htm](https://www.glassdoor.com/index.htm)
- [https://www.careercup.com/](https://www.careercup.com/)


## 基础知识
- 垃圾回收机制gc
- 全局锁GIL
- python里的list和dict是怎么实现的
- python里的with怎么实现的


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


### 操作系统
- [Operating Systems and System Programming - UC Berkeley CS 162](https://github.com/Berkeley-CS162)
- [Operating System Engineering - MIT 6.828](https://pdos.csail.mit.edu/6.828/)
- [编码：隐匿在计算机软硬件背后的语言](https://book.douban.com/subject/4822685/)
- [计算机系统要素](https://book.douban.com/subject/1998341/)
- [计算机组成与设计](https://book.douban.com/subject/26604008/)
- [Operating Systems: Principles and Practice](https://book.douban.com/subject/25984145/)
- [Operating Systems: Three Easy Pieces](https://book.douban.com/subject/19973015/)
- [Operating System Concepts](https://book.douban.com/subject/10076960/)


### Web开发与系统设计
- [Introduction to Computer Networking - Stanford](https://lagunita.stanford.edu/courses/Engineering/Networking-SP/SelfPaced/about)


### 分布式
- [Distributed Systems - MIT 6.824](https://pdos.csail.mit.edu/6.824/schedule.html)
- [Talent Plan | PingCAP University](https://university.pingcap.com/talent-plan/)
- [数据密集型应用系统设计](https://book.douban.com/subject/30329536/)
