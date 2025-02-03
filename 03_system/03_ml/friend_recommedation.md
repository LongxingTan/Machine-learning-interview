# 可能认识的人推荐

> People You May Know: u2u推荐


## 1. requirements
**场景/产品**
- Symmetrical Friendship: The system should recommend connections where both users are likely to accept the friendship.

**目标**
- Network Growth: help users discover potential connections
- User Retention: Improve user retention by enhancing the social experience

**约束**
- Scale: the total number of users on the platform, and daily active users
- Latency: Recommendations should be generated in real-time (e.g., < 200ms) to ensure a smooth user experience
- average connections for one user


## 2. ML task & pipeline
- 利用共同好友、位置、教育背景、工作经历判断可能认识
- 输入用户信息，输出和该用户最相似的k个用户作为推荐

![](../../.github/assets/03ml-friend-pipe.png)


## 3. data
- User Profiles: Demographics, interests, and preferences
- Social Graph: Existing connections and interactions (e.g., mutual friends, common groups)
- Behavioral Data: Past interactions (e.g., profile views, connection requests)



## 4. feature
**User Features**
- Demographics: Age, location, education
- Interests: Hobbies, favorite topics
- Behavior: Frequency of interactions, types of interactions

**Connection Features**
- Common Friends: Number of mutual friends. 共同好友个数
- Shared Interests: Overlap in interests or groups
- Interaction History: Past interactions between users (e.g., profile views, messages)

**Context Features**
- Time of Day: Recommendations may vary based on the time of day
- Device: Recommendations may differ based on the device used (e.g., mobile vs. desktop)


## 5. model

**ranking**

![](../../.github/assets/03ml-friend-rank.png)

Collaborative Filtering: Recommend connections based on similar users' connections.

point-wise learning to rank
- task 2 users as input, output the probability of forming a friend

graph based prediction
- graph level prediction
  - predict if a chemical compound us an enzyme
- node level prediction
  - predict if a specific user is a spammer
- edge level prediction
  - predict if two users likely to connect


## 6. evaluation
**offline**
- precision, recall, and AUC-ROC

**online**
- A/B Testing: Continuously test new models and features to measure their impact on engagement and network growth
- Feedback Loop: Use user feedback to retrain and improve models


## 7. deployment & serving
- 部分 batch serving


## 8. monitoring & maintenance


## reference
- [大规模异构图召回在美团到店推荐广告的应用](https://tech.meituan.com/2022/11/24/application-of-large-scale-heterogeneous-graph-in-meituan-recommended-ads.html)
- [推荐系统u2u算法简介 - Shard Zhang的文章 - 知乎](https://zhuanlan.zhihu.com/p/665867589)
- [people you may know](https://webupon.com/blog/linkedin-people-you-may-know-algorithm/)
