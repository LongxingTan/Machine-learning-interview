# 产品分析

对于数据科学家，需要对产品案例分析刷题和总结。

- 注意framework
- 听清楚考官在问什麽。不要一上来就ab test，quasi-experimentation
- 提出你有的疑问，质疑每一句考官问你的话，不要他说什麽你都接受
- 考官问完后，请你跟他说你需要一点时间，统整你的思路跟回答
- 把自己想成该产品的owner
- 每一项回答，背后都要有“理由”。为什麽这样，不那样？为什么选这个metric,不是其他？
- 每一项回答，都想清楚背后的取舍。把 trade-off讲清楚, 是能否拿到senior的关键。Junior pursues right or wrong; Senior looks for trade-offs


## 1. 范围

结合纵向和横向去做题目分类总结。例如，我们模拟一下FB groups 可以大致问：
- 我们想要新增某个feature 让用户可以在回文的时候使用表情包，该不该做？
- 我们想要improve comments per post, 该怎麽做？
- 我们看到comments per post下降了，该怎麽办
- 我们想要build a model 让FB groups的贴文出现在个人的newsfeed, 该怎麽做？

### 1.1 纵向-题目类型
- Launch or not: 公司想要针对产品某部分优化，开发某种新功能，该不该做？
- Investigation: 某个产品的表现不如预期，某指标下降，该怎麽调查釐清？
- How to measure: 如何衡量某个产品的好坏
- Want to know something: 想要了解某件事情，例如有多少百分比的帐号是假帐号，怎麽知道有人有多个帐号？
- How to build a model: Meta的某些产品题目是直接涉及到model building, 例如餐厅推荐，侦测假帐号。这种题目较为niche, 但仍要准备。

### 1.2 横向-产品线
- 电商 （FB/ IG shop)
- FB Groups
- FB Newsfeed
- FB ads
- Messenger


## 2. 回答框架Framework

### 2.1 目标
以framework来说，第一步，也是最重要的，是订所谓的目标。目标有两种
- 整个公司有一个大的overall business goal. 例如meta的愿景是让人们之间的距离缩短，并让人们有能力可以打造社群，也帮助企业成长盈利。这不是我发明的，是他们家的mission statement
- 题目中涉及的产品也会有一个主要目标。例如增加某feature engagement

所有你提出的策略，跟产品的目标，再到公司的大目标，彼此都是环环相扣。
在准备不同公司的时候，都仔细想想这家公司是做什麽的，有哪些产品，然后请你产生出自己对于该公司独家的framework.

FB mission (business goal) → Product goal (pain point solving / why are we doing this) → Hypothesis → Validation (Metrics and methodology i.e experiment design) → Analysis → Decision
先从FB的愿景讲起--> 然后自定义产品的目标为何，这个目标也是要能够帮助到整体企业的目标--> 有哪些假说 → 要如何确定假说，metric and methodology (i.e experiment) → 如何分析结果--> 结果为如何的时候我们怎麽下决策

### 2.2 metrics
讲metric 的时候，要跟目标有所结合。不要随便乱丢metric
Metrics有分成几种
- Goal (success) metrics: 你想要提升的目标
- Monitoring metrics
- Guardrail metrics: 不能看到显着影响的指标

每个metric都有pros and cons. 我认为常见的trade off 有
- Engagement v.s monetization (有时增加互动，但短期内的收益会减少）
- Short term v.s long term （有些指标无法反映长期）
- Engagement v.s safety （互动增加但可能伤害人与人之间的互动，例如耸动的新闻或假新闻）

### 2.3 insight analysis

提出假设

### 2.4 实验
请把A/B test的架构讲得清楚:
- 实验组跟对照组各可以干嘛
- Randomization unit. 为何选择这个，而不是另一个？最常回答的用user_id 来当作randomization unit,会有什麽pros and cons?
- 如果A/B test 不可行的时候，该怎麽办？为什麽不可行？
- 如何 identify network effect? How to mitigate the risk?


## 3. 案例- Analytical Reasoning: Restaurant Recommendations

FB在考虑build一个餐厅推荐system，插入到user的news feed里面
How would you decide if this might be worth while？
- 大概就是问opportunity sizing，要pull什么data之类的

How would you design the first iteration of the model?
- 我回答logistical model 然后input可以用user的activity history，location，他们friends的activity
- 如果没有这些data，可以先推荐local popular restaurants

How would you validate your model is working?.
- 我说可以用A/B test然后看how our key metrics change in the two groups
- 另外可以自己抽样，看看recommendation是不是make sense，我们是不是落了什么factor

What would you do if ads revenue from restaurants increase 5% but engagement down 3%?
- 我就是说先确定这两个是不是有联系，再segment到不同的region和demographic看有没有specific，如果有specific可以看看是哪里出了问题，是不是有cultural difference，如果是的话可以根据那个design一版custom的

When would you decide the time to ingest into newsfeed?
- 我说是可以看user有没有固定时间每天用这个app的
- 另外可以看这个user他如果有很多要看的post，那就先prioritize post；如果他本来每天就能看完，那就可以prioirtize推荐餐厅


前提是Advertiser在fb上买广告，假设target audience size M, purchased N impression
1. Probability an individual see at lease one impression
2. Expected value of total people who see at least one impression
3. We’ve run a prediction model and discovered 25% of our audience is high intent (90% probability of clicks) and 75% are low intent (10% clicks), how many clicks do we expect to see?
4. If the advertiser are concerned of 0 clicks and want to increase the number of impression they buy. X axis is number of impressions purchased and Y is likelihood of getting 0 clicks, how does X and Y change (draw graph)


## 参考
- Ace the data science interview
- [Meta Senior DS, Product Analytics 面试准备总结](https://www.1point3acres.com/bbs/thread-1012204-1-1.html)
- [new grads湾区DA/DS找工作超细致回顾+面经+资料总结](https://www.1point3acres.com/bbs/thread-469408-1-1.html)
- [数据科学家面试 Data Scientist Interview Product sense/metrics 套路总结](https://www.1point3acres.com/bbs/thread-679303-1-1.html)
- [字节跳动-AI数据分析](https://www.1point3acres.com/bbs/thread-1028399-1-1.html)
- [Meta DSA 全套过经+Timeline](https://www.1point3acres.com/bbs/thread-1042322-1-1.html)
- [推荐系统实用分析技巧 - 纳米酱的文章 - 知乎](https://zhuanlan.zhihu.com/p/188228577)
- [allocation budget](https://blogboard.io/blog/data-science-in-marketing-optimization/)
- [https://medium.com/stellarpeers](https://medium.com/stellarpeers)
- [https://www.tryexponent.com/questions?page=1&type=product-design](https://www.tryexponent.com/questions?page=1&type=product-design)
- [24年初 DS 跳槽两个月拿到四个 offer 经验总结](https://www.1point3acres.com/bbs/thread-1058665-1-1.html)
- [product sense解题思路以及数据分析面试资料分享](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=806048&ctid=229383)
