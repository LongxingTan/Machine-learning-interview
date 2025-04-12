# 爬虫

A web crawler that will crawl Wikipedia

- We need to deploy the same software on each node. We have 10,000 nodes, the software can know about all the nodes. We have to minimize communication and make sure each node does equal amount of work.
- 有一些constraint，需要跟面试官交流。给你一千个bot做web crawler，不重复crawl，bot之间不talk

## 代码版

- you are given 3 APIs: fetch (given a url, downloads its content), parse (given the downloaded content, get all urls inside the content), save (dump the content to disk)

单线程

```python

```

多线程

```python

```

## requirements

## Reference

- https://leetcode.com/discuss/interview-question/system-design/124657/Face
