# News feed

News is an inherently diverse category, spanning various topics and catering to a wide array of user types and personas, such as adults, business professionals, general readers, or specific cohorts with unique interests

> personalization accuracy, 时效性

## 1. requirements

**场景/功能类**

- 是否涉及到不同user的networking
- 是否需要热点推荐: 及时发现热点事件， 并挖掘出来对应报道，才能有效帮助热点进行推荐和分发。
- Do we show only posts or also activities from other users?
- What types of engagement are available? (like, click, share, comment, hide, etc)? Which ones are we optimizing for?
- Do we display ads as well?
- Are there specific user segments or contexts we should consider (e.g., user demographics)?
- Do we have negative feedback features (such as hide ad, block, etc)?
- What type of user-ad interaction data do we have access to can we use it for training our models?
- How do we collect negative samples? (not clicked, negative feedback).

**目标类**

- What is the primary business objective of the system?
  - increase user engagement: clicks, time spent, shares
- 多样性(Diversity)
- Freshness

**约束类/non-functional**

- What types of data do the posts include?
  - text, image, video
- What is the scale of the system?
  - millions of users and thousands of articles per second
- How fast the system needs to be?
  - < 200ms

## 2. ML task & pipeline

a default recommendation system based on their location and country can serve as a starting point. However, the primary focus should be on gathering signals from returning users to refine the algorithm's understanding of their preferences and homepage experience. This could include monitoring how much time they spend on specific news categories and soliciting feedback. The feedback mechanism should give users the option to indicate their preferences, such as whether they appreciate a particular recommendation or would rather avoid certain types of news. Given the sensitivity of news topics, especially in areas like war or other contentious subjects, providing users with the option to opt out of future recommendations in such categories is crucial.

By observing user behavior, such as the articles they read and the time they spend on them, the algorithm can suggest related articles and gauge user engagement. The 'You May Also Like' (YMAL) approach, based on users' past reading history, is another valuable recommendation strategy. It suggests content similar to what users have previously engaged with, enhancing the overall user experience

**trade-off**

## 3. data collection

**user**

- Demographics, past interactions (clicks, likes, shares), and social graph data (if applicable)

**item**

- Articles, posts, videos, and images

**engagement**

- Positive (clicks, likes) and negative (hides, blocks) feedback

**context**

## 4.feature

## 5. model

**retrieval**: multiple channel retrieval

- Collaborative Filtering
- Content-Based Filtering

**ranking**: multiple object ranking

- GBDT
- NN

## 6. evaluation

- offline
- online: A/B testing
  - DAU和留存
  - CTR, spend time

## 7. deployment & serving

- batch serving
- online serving

## 8. monitor & maintenance

- Do we need continual training?

## 9. 问答&优化

### 9.1 bias

- position bias

### 9.2 cold start

it's crucial to establish a robust onboarding process (capturing onboarding signals). This process aids in understanding and improving the user's homepage experience.
When a user logs into your service for the first time, gathering fundamental information is invaluable. This includes their areas of interest, the average time they spend on news consumption, and whether they aim to enhance their reading habits daily.
Additionally, asking users to list the specific topics they are interested in can be a game-changer in addressing the 'cold start' problem, which occurs when users provide minimal initial interaction data.

### 9.3 混排

- 多题材混排
  - 先各自排序，再混排
  - listwise

## reference

**精读**

- [How machine learning powers Facebook’s News Feed ranking algorithm](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/)
- [Serving Billions of Personalized News Feeds with AI - Meihong Wang (Facebook)](https://www.youtube.com/watch?v=wcVJZwO_py0&t=80s)

**扩展**

- [Powered by AI: Instagram’s Explore recommender system](https://ai.meta.com/blog/powered-by-ai-instagrams-explore-recommender-system/)
- [Lessons Learned at Instagram Stories and Feed Machine Learning](https://instagram-engineering.com/lessons-learned-at-instagram-stories-and-feed-machine-learning-54f3aaa09e56)
- [Community-focused Feed optimization](https://engineering.linkedin.com/blog/2019/06/community-focused-feed-optimization)
- [Building a dynamic and responsive Pinterest](https://medium.com/pinterest-engineering/building-a-dynamic-and-responsive-pinterest-7d410e99f0a9)
- [交互式推荐在外卖场景的探索与应用](https://mp.weixin.qq.com/s/s7yoJXgc_7txSooeuE-3sg)
- [深度召回在飞猪旅行推荐系统中的探索和实践](https://mp.weixin.qq.com/s/AyMmfixX8rXUgGIf94uBkw)
- [Personalized News Recommendation: Methods and Challenges](https://arxiv.org/pdf/2106.08934)
- [https://github.com/datawhalechina/fun-rec](https://github.com/datawhalechina/fun-rec)
- [A Dive into the Facebook Newsfeed Architecture](https://algodaily.com/lessons/dive-into-facebook-newsfeed-architecture)
