# 去重复性检测/版权检测

## 1. requirements

- 机器学习系统中处理相似商品推荐时，去重复（deduplication）可以确保用户获得多样化且相关的商品选项
  - 商品推荐：检测商品是否为重复条目（例如不同商家上传了相同的商品图片或描述）
  - 视频版权检测：检测用户上传的视频是否与现有视频库中的内容重复，以保护版权
- user上传的视频是否和a large media collection里的视频有重复


能够高效处理大规模数据（例如：数百万条商品或视频）
支持近实时检测（例如：商品上传或视频上传后快速完成重复性判断）
系统需兼顾计算效率和检索准确率


## 2. ML task & pipeline
- 基本属性来检测重复(ID, 相似度)
- 局部敏感哈希, video做hashing，用bloom filter
- 做embedding，放vector database，找nearest neighbor


## 3. data collection
商品去重：
- 商品图片、标题、描述文本等
- 数据来源于电商平台上的商品库

视频版权检测：
- 用户上传的视频
- 已知的版权视频库（large media collection）


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
- [美团外卖基于GPU的向量检索系统实践](https://tech.meituan.com/2024/04/11/gpu-vector-retrieval-system-practice.html)
- [MIND TensorFlow serving 部署预测用户embedding - 亓逸的文章 - 知乎](https://zhuanlan.zhihu.com/p/486282241)
- [Using machine learning to index text from billions of images](https://dropbox.tech/machine-learning/using-machine-learning-to-index-text-from-billions-of-images)
