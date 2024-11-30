# POI推荐

> 场景: yelp, 美团, airbnb
> - Design a system to find nearby restaurants
> - Design a system to match drivers with riders for Uber
> - Design a system to compute ETA for food delivery

> 特点: 如果是event 推荐这种注重实效性、位置性的推荐，event发生后不存在了，所有item可以认为都是冷启动
> 对于位置的挖掘可采用图特征或模型


## 1. requirements
**products/use cases**

**objective**
- connect people with great local businesses

**constraint**
- data
- volume
- latency


## 2. ML task & pipeline

预测目标
- 是否点击
- 停留时间(dwell time), 可转化为t/(t+1)来逼近sigmoid函数，t很大时接近1；很小时接近0


## 3. data collection
- user
  - User location: For localized recommendations we need to consider only businesses near the city or neighborhood where the user is located


## 4. feature
- sparse
- dense


## 5. model
**retrieval**
- 取决于filter

**ranking**

**rerank**


## 6. evaluation
- offline
  - NDCG
  - MAP
- online: A/B testing holdout canary


## 7. deploy & serving
- batch serving
- online serving


## 8. monitor & maintenance


## 9. 优化与问答
冷启动的item
- 双塔可以采用default embedding, 而不是random initial


## reference
- [yelp-Beyond Matrix Factorization: Using hybrid features for user-business recommendations](https://engineeringblog.yelp.com/2022/04/beyond-matrix-factorization-using-hybrid-features-for-user-business-recommendations.html)
- [Yelp Food Recommendation System](https://cs229.stanford.edu/proj2013/SawantPai-YelpFoodRecommendationSystem.pdf)
- [美团-旅游推荐系统的演进](https://tech.meituan.com/2017/03/24/travel-recsys.html)
- [美团-基于机器学习方法的POI品类推荐算法](https://tech.meituan.com/2014/12/18/poi-category-recommendation-algorithm-based-on-machine-learning.html)
- [Design Yelp](https://systemdesignschool.io/problems/yelp/solution)
- [System Design — Nearby Places Recommender System](https://mecha-mind.medium.com/system-design-nearby-places-recommender-system-7ac53e27c977)