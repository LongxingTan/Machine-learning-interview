# Log System

> a distributed data store for logs

## 1. requirements

- 要求能够将PII信息隐藏，能够做top k exceptions，能够做快速查询，比如按各种不同的fi
- Our goal is to build a real-time aggregation system. WebServer logs event to our system using the following function call: log(ad_id, user_id, event_type).
  At this moment we plan to have only 2 types of events: "ad was shown" and "ad was clicked on".

## 参考

- https://engineering.fb.com/2017/08/31/core-data/logdevice-a-distributed-data-store-for-logs/
