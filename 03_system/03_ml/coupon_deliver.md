# 优惠券发放

优惠券可以在特定场景下刺激消费者冲动消费，从而实现利润最大化

## 1. requirements
Primary Goal: Maximize overall profit by:
- Increasing purchase frequency.
- Boosting average order value (AOV).
- Encouraging purchases of high-margin items.

Secondary Goals:
- Improve customer retention and loyalty.
- Minimize wasted distribution of coupons (cost efficiency).
- Balance supply and demand (e.g., avoid overloading inventory).

**Constraints**
- Budget Limit: Total cost of distributed coupons must not exceed a predefined budget.
- Customer Segmentation: Different coupons for different user groups (e.g., high-value vs. low-value users).
- Redemption Rate: Target a specific redemption rate to ensure coupon effectiveness.
- Coupon Fatigue: Avoid over-distributing coupons to prevent diminishing returns or customer dissatisfaction.
- Scalability: Handle millions of users and products.


## 2. task & pipeline
>  机器学习+运筹: hybrid approach combining machine learning for prediction and operations research for optimization

key challenge:
- Imbalanced Data
- Cold Start
- Scalability


## reference
- [大厂的优惠券系统是如何设计的？ - JavaEdge的文章 - 知乎](https://zhuanlan.zhihu.com/p/511822092)
- [如何设计优惠券系统？一文带你看懂优惠券 - 薛老板的文章 - 知乎](https://zhuanlan.zhihu.com/p/351658623)
