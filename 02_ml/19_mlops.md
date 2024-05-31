# MLOPS
工业部署模型需要掌握的相关知识

## ML部署

- tf-serving
  - 支持热部署，不会使服务失效

- flask
  - 压力测试 jmeter


- 模型
  - an end-to-end set
  - a confidence test set
  - a performance metric
  - its range of acceptable values

- Recovery


- Serving in Batch Mode

- 量化


- 高性能
  - C++重写inference，配上模型加速措施(剪枝，蒸馏，量化)，高并发请求


- LLM推理
  > fast-transformer, vllm等框架
  - attention: flash attention, paged attention
  - MOE

- gpu多实例部署


## 模型压缩

- 蒸馏
  - 如何设计合适的学生模型和损失函数

- 量化
  - 减少每个参数和激活的位数（如32位浮点数转换为8位整数)，来压缩模型的大小和加速模型的运算

- 低秩分解近似


## retrain
> develop a strategy to trigger model invalidations and retrain models when performance degrades.
> because of data drift, model bias, and explainability divergence


## 问答
- 模型部署后，怎么检测模型流量


## 参考
- [python实时语音识别服务部署 - 叫我小康的文章 - 知乎](https://zhuanlan.zhihu.com/p/467364921)
- [通用目标检测开源框架YOLOv6在美团的量化部署实战](https://tech.meituan.com/2022/09/22/yolov6-quantization-in-meituan.html)
- [炼丹师的工程修养之五：KubeFlow介绍和源码分析](https://zhuanlan.zhihu.com/p/98889237)
- 模型推理服务化框架Triton
- https://github.com/rapidsai/cloud-ml-examples
- [模型部署优化学习路线是什么？ - Leslie的回答 - 知乎](https://www.zhihu.com/question/411393222/answer/2359479242)
- [Version and track Azure Machine Learning datasets](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-version-track-datasets?view=azureml-api-1)
-
