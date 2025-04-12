# 设计Youtube

## 视频存储

- 海量用户 海量视频
- 视频文件：distributed file system
  - 热门视频： CDN缓存
  - 分片上传：拆分成小视频
  - 上传结束后：不同分辨率保存不同版本
- 视频meta data: relational database

- 读：热门视频redis缓存，冷门去hive读出来再缓存
- 写：先写缓存，异步写入hive，同时上传

## Reference

- [Scaling Facebook Live Videos to a Billion Users](https://www.youtube.com/watch?v=IO4teCbHvZw)
