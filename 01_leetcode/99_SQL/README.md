# SQL

> 建议下载安装一个My SQL，模拟真实SQL环境学习，尤其数据科学岗位
>
> - 对product和metrics的业务了解能够帮助我们更好的准备SQL面试，SQL面试都是围绕着**与business相关的metrics**展开。
> - [LeetCode SQL Top 50](https://leetcode.com/studyplan/top-sql-50/)

## 知识点

- 书写顺序: SELECT -> DISTINCT -> FROM -> JOIN -> ON -> WHERE -> GROUP BY -> HAVING -> ORDER BY -> LIMIT
- 执行逻辑顺序: FROM -> JOIN -> ON -> WHERE -> GROUP BY -> HAVING -> SELECT -> DISTINCT -> ORDER BY -> LIMIT
- self-join
- case when
- aggregate function: SUM(), MAX(), MIN(), AVG()
- WITH common_table_expression
- explode
- Windows function: `<窗口函数> over (partition by (分组的列名) order by (排序的列名)) AS name`
  - 解决组内排名问题，top N：找出每个部门排名前 N 的员工进行奖励
  - group by 分组汇总后改变了表的行数，一行只有一个类别; partition by 和 rank 函数不会减少原表中的行数
  - row number / rank / dense_rank
- frame clause
- NULL value
- TIMESTAMP
- 数据库优化

  - 创建并使用正确的索引
  - 只返回需要的字段
  - 减少交互次数（批量提交）
  - 设置合理的Fetch Size（数据每次返回给客户端的条数）
  - EXPLAIN 关键字 是 MySQL 中用来分析和优化 SQL 查询的重要工具

- where和having之后都是筛选条件

  - where在group by前， having在group by 之后
  - 聚合函数（avg、sum、max、min、count），不能作为条件放在where之后，但可以放在having之后

- Hive和MySQL
  - Hive是基于Hadoop的数据仓库, 能处理 PB 级别的数据, HDFS存储数据, 数据模型列存储, 批处理和分析型查询
  - MySQL关系型数据库, 行式存储模型, 实时查询和小规模数据处理

## 常见问题

- What is the difference between union and union all? where and having?
- List out the top 3 names of the users who have the most purchase amount on '2018-01-01'
- Sort the table by timestamp for each user. Create a new column named "cum amount" which calculates the cumulative amount of a certain user of purchase on the same day.
- For each day, calculate the growth rate of purchase amount compared to the previous day. if no result for a previous day, show 'Null'.
- For each day, calculate a 30day rolling average purchase amount.
- What was the friend request acceptance rate for requests sent out on 2018-01-01?

## 实际

关键配置参数

- innodb_buffer_pool_size、sync_binlog、innodb_log_file_siz
- SQL explain 优化
- python\scala 三个连续双引号表示跨行字符串，`.format()`
- Change Data Capture
  - 跟踪和捕获数据库中数据变更，变更数据同步到其他系统（如数据仓库、缓存、搜索索引等）
- ORM

## Reference

- [SQL ZOO](https://www.sqlzoo.net/wiki/SQL_Tutorial)
- [w3schools.com/sql](https://www.w3schools.com/sql/)
- https://github.com/oleg-agapov/data-engineering-book/blob/master/book/2-beginner-path/2-2-sql-for-beginners/sql-1.md
- 18 best sql online learning resources
- [数据分析人员需要掌握sql到什么程度？ - 无眠的回答 - 知乎](https://www.zhihu.com/question/379694223/answer/1118850805)
- [OLAP入门问答-进阶篇](https://zhuanlan.zhihu.com/p/147344996)
- [Hive 优化](https://zhuanlan.zhihu.com/p/102475087)
- [https://github.com/siqichen-usc/LeetCode-SQL-Summary](https://github.com/siqichen-usc/LeetCode-SQL-Summary)
- Mosh SQL course
- [platform.stratascratch.com](platform.stratascratch.com)
- [SQL 窗口函数 ( window function ) - 子皿三吉的文章 - 知乎](https://zhuanlan.zhihu.com/p/390381181)
- udemy: SQL - MySQL for Data Analytics and Business Intelligence
- The Ultimate MySQL Bootcamp
- sqlbolt.com
- mode.com/sql-tutorial
- sqlteaching.com
- selectstarsql.com
