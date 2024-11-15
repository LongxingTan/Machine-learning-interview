# SQL

数据科学岗位，需要好好刷SQL题。建议自己下载一个My SQL装到电脑上，模拟真实SQL环境来学习。

对product和metrics的业务了解能够帮助我们更好的准备SQL面试，SQL面试都是围绕着**与business相关的metrics**展开。


## 知识点
- 书写顺序: SELECT -> DISTINCT ->  FROM -> JOIN -> ON ->  WHERE -> GROUP BY -> HAVING -> ORDER BY -> LIMIT
- 执行逻辑顺序: FROM -> JOIN -> ON -> WHERE -> GROUP BY -> HAVING -> SELECT -> DISTINCT -> ORDER BY -> LIMIT
- row number
- explode
- Windows function
- frame clause
- self-join
- case when
- aggregate function
- WITH common_table_expression
- rank
- 数据库优化
  - 创建并使用正确的索引
  - 只返回需要的字段
  - 减少交互次数（批量提交）
  - 设置合理的Fetch Size（数据每次返回给客户端的条数）

- where和having之后都是筛选条件
  - where在group by前， having在group by 之后
  - 聚合函数（avg、sum、max、min、count），不能作为条件放在where之后，但可以放在having之后


## 常见问题
- What is the difference between union and union all? where and having?
- Question 1: List out the top 3 names of the users who have the most purchase amount on '2018-01-01'
- Question 2: Sort the table by timestamp for each user. Create a new column named "cum amount" which calculates the cumulative amount of a certain user of purchase on the same day.
- Question 3: For each day, calculate the growth rate of purchase amount compared to the previous day. if no result for a previous day, show 'Null'.
- Question 4: For each day, calculate a 30day rolling average purchase amount.
- Question: what was the friend request acceptance rate for requests sent out on 2018-01-01?


## 实际
关键配置参数
- innodb_buffer_pool_size、sync_binlog、innodb_log_file_siz
- SQL explain 优化
- python\scala 三个连续双引号表示跨行字符串，`.format()`


## Reference
- SQL ZOO 
- w3schools.com/sql
- udemy: SQL - MySQL for Data Analytics and Business Intelligence 和The Ultimate MySQL Bootcamp
- sqlbolt.com
- mode.com/sql-tutorial
- Hackerrank
- sqlteaching.com
- selectstarsql.com
- https://github.com/oleg-agapov/data-engineering-book/blob/master/book/2-beginner-path/2-2-sql-for-beginners/sql-1.md
- 18 best sql online learning resources
- [数据分析人员需要掌握sql到什么程度？ - 无眠的回答 - 知乎](https://www.zhihu.com/question/379694223/answer/1118850805)
- https://zhuanlan.zhihu.com/p/147344996
- [Hive 优化](https://zhuanlan.zhihu.com/p/102475087)
- [https://github.com/siqichen-usc/LeetCode-SQL-Summary](https://github.com/siqichen-usc/LeetCode-SQL-Summary)
- Mosh SQL course
- [platform.stratascratch.com](platform.stratascratch.com)
