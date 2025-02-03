# POI推荐

> 场景: yelp, 美团, airbnb
> - Design a system to find nearby restaurants
> - Design a system to match drivers with riders for Uber
> - Design a system to compute ETA for food delivery

> 特点: 如果是event 推荐这种注重实效性、位置性的推荐，event发生后不存在了，所有item可以认为都是冷启动
> 对于位置的挖掘可采用图特征或模型


## 1. requirements
**products/use cases**
- User Search: Users search for restaurants based on location, cuisine, and preferences.
- Real-Time Recommendations: Provide real-time recommendations based on user queries.
- Cold Start: Handle new restaurants with limited data.

**objective**
- Connect Users with Local Businesses: Help users discover great local businesses
- Increase Engagement: Encourage users to explore and interact with more POIs.

**constraint**
- Data Constraints: Limited data on new restaurants.
- Volume: Handle a high volume of users and queries.
- Latency: Provide real-time results with low latency (e.g., < 200ms).


## 2. ML task & pipeline

预测目标
- 是否点击
- 停留时间(dwell time), 可转化为t/(t+1)来逼近sigmoid函数，t很大时接近1；很小时接近0


## 3. data
**Data collection**
- User Profiles: Demographics, preferences, and past interactions
- POI Data: Location, cuisine, ratings, reviews, and other attributes
  - User location: For localized recommendations we need to consider only businesses near the city or neighborhood where the user is located
  - Business Data: Restaurant location, cuisine type, user ratings, and reviews
- Interaction Data: Past searches, clicks, and visits

**Data Processing**
- Data Cleaning: Handle missing data and outliers
- Data Integration: Combine data from different sources into a unified format
- Data Augmentation: Use techniques like synthetic data generation to handle cold start problems


## 4. feature
**sparse**

**dense**

**User Features**
- Demographics: Age, location
- Preferences: Favorite cuisines, price range
- Behavior: Past searches, clicks, and visits

**POI Features**
- Location: Latitude, longitude, and proximity to the user
- Attributes: Cuisine, price range, ratings, reviews
- Popularity: Number of visits, ratings, and reviews

**Context Features**
- Time of Day: Recommendations may vary based on the time of day
- Device: Recommendations may differ based on the device used (e.g., mobile vs. desktop)


## 5. model

**retrieval**
- 取决于filter
- Collaborative Filtering: Recommend POIs based on similar users' preferences
- Content-Based Filtering: Recommend POIs similar to those the user has interacted with in the past
- Graph-Based Models: Use graph algorithms (e.g., Node2Vec, GraphSAGE) to capture spatial relationships between POIs


**ranking**


**rerank**


## 6. evaluation
- offline
  - NDCG
  - MAP
  - precision, recall, and AUC-ROC
- online: A/B testing holdout canary


## 7. deploy & serving
- Batch Serving: Periodically update restaurant recommendations.
- Online Serving: Real-time requests for user queries.

## 8. monitor & maintenance


## 9. 优化与问答
- 冷启动的item
  - 双塔可以采用default embedding, 而不是random initial


## reference
- [yelp-Beyond Matrix Factorization: Using hybrid features for user-business recommendations](https://engineeringblog.yelp.com/2022/04/beyond-matrix-factorization-using-hybrid-features-for-user-business-recommendations.html)
- [Yelp Food Recommendation System](https://cs229.stanford.edu/proj2013/SawantPai-YelpFoodRecommendationSystem.pdf)
- [美团-旅游推荐系统的演进](https://tech.meituan.com/2017/03/24/travel-recsys.html)
- [美团-基于机器学习方法的POI品类推荐算法](https://tech.meituan.com/2014/12/18/poi-category-recommendation-algorithm-based-on-machine-learning.html)
- [Design Yelp](https://systemdesignschool.io/problems/yelp/solution)
- [System Design — Nearby Places Recommender System](https://mecha-mind.medium.com/system-design-nearby-places-recommender-system-7ac53e27c977)