# LeetCode 面试准备指南

> LeetCode在面试时常被认为是衡量CS水平的重要指标。**精刷大于泛刷，熟练度大于题量**。因此，尽量提高熟练度，并踩对得分点和signal

## 1. 面试评分维度
> 大厂面试相对规范和模版化。评分细则化，想到某个方向加多少分，写一个实现加多少分，有bug扣多少分，答错复杂度扣多少分等。

大厂面试通常从四个维度评分：
- Communication: 沟通表达能力
- Problem Solving: 问题分析能力
- Coding: 代码实现能力
- Verification: 代码验证能力

## 2. 刷题策略
> 快速给出bug free代码，有良好的**沟通**和**思路介绍**。做题前项目介绍不必随意展开(1~2min)，留够做题时间。

### 2.1 平时准备
1. **基础准备**：掌握常见数据结构和算法，熟悉各类题型的解题模板
2. **系统训练**：按tag刷题，强化模板应用；尝试多种解法，提高熟练度
3. **实战演练**：参加周赛，提升实战能力；练习边说边写，模拟面试场景
4. **针对性准备**：根据目标公司调整刷题重点；找最新面经，针对性练习

### 2.2 面试解题流程

1. **理解题目**
   - 仔细阅读题目要求
   - 提出 clarifying questions。可以通过列举test case，确保理解问题并涵盖所有情况
   - 确认输入输出类型和边界条件 (input/output type)
     - 输入数字全是正数或全是负数，会不会是null or empty
     - 输入string是不是都是English letters (A-Z)，有没有特殊符号
     - 输入是否有重复样本，输出是否允许重复
     - standard input, empty input, malformed input, long input, null input, other edge input
     - 能不能改input，immutable or not (1D/2D array比较常见）
     - 特殊情况输出返回什么
     - 要不要in place
     - 有没有内存限制
2. **思路分析**
   - 与面试官讨论解题思路
     - Whenever coming across a solution, talk it out and discuss it
   - 确认思路可行性，思路未得到认同前不要急于写代码
   - 必要时寻求提示
3. **代码实现**
   - 选择合适的数据结构并介绍
   - 实现算法
   - 处理边界情况
4. **验证优化**
   - 编写测试用例
   - 分析时间和空间复杂度
   - 讨论可能的优化方案

## 3. 公司特点

### Google
- 注重问题解决过程
- 原题较多，难度较高
- 重视算法思维和测试用例设计

### Meta
- 35分钟两道题
- 重视代码质量和最优解
- ML system design 重点：推荐系统、视频搜索

### Microsoft
- 基础算法和数据结构
- 注重代码实现速度

### Amazon
- Grind75 题目为主
- 15分钟一道题
- BQ 占比较大（30分钟+）

### Tiktok
- 题目不按 tag
- ML 方向注重项目深度

## 4. 学习资源

### 4.1 刷题列表
- [Blind 75](https://leetcode.com/list/xi4ci4ig/)
- [Neetcode 300](https://neetcode.io/practice)
- [代码随想录](https://programmercarl.com/)
- [LeetCode 101](https://github.com/changgyhub/leetcode_101/)
- [灵茶山艾府](https://github.com/EndlessCheng)

### 4.2 编程语言基础
- Python: GC机制、GIL、数据结构实现
- Java: [Java Guide](https://javaguide.cn/home.html)
- C++: 容器库、并发编程

### 4.3 计算机基础
- CS50x: 计算机科学导论
- CS61A: 程序结构与解释
- CSAPP: 深入理解计算机系统
- 算法导论
- 分布式系统 (MIT 6.824)
- 操作系统 (MIT 6.s081)
- 数据库 (CMU 15445)
