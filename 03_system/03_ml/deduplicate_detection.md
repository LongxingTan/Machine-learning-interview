# 去重复性检测

## 1. requirements

- 机器学习系统中处理相似商品推荐时，去重复（deduplication）可以确保用户获得多样化且相关的商品选项
- user上传的视频是否和a large media collection里的视频有重复


## 2. pipeline
- 基本属性来检测重复(ID, 相似度)
- 局部敏感哈希, video做hashing，用bloom filter
- 做embedding，放vector database，找nearest neighbor


## 3. data collection
