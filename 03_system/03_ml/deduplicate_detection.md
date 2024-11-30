# 去重复性检测/版权检测

## 1. requirements

- 机器学习系统中处理相似商品推荐时，去重复（deduplication）可以确保用户获得多样化且相关的商品选项
- user上传的视频是否和a large media collection里的视频有重复


## 2. ML task & pipeline
- 基本属性来检测重复(ID, 相似度)
- 局部敏感哈希, video做hashing，用bloom filter
- 做embedding，放vector database，找nearest neighbor


## 3. data collection


## 4. feature
- NN


## 5. model


## 6. evaluation


## 7. deploy & serving


## 8. monitoring & maintenance


## Reference
- [Introduction to Facebook AI Similarity Search (Faiss)](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)
- [AutoConsis：UI内容一致性智能检测](https://mp.weixin.qq.com/s/VwnnYnyo9sCDdUuG4Mu1kQ)
- [How image search works at Dropbox](https://dropbox.tech/machine-learning/how-image-search-works-at-dropbox)