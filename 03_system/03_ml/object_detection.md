# 目标检测

## 1. requirements

**Functional Requirements**

- 检测对象 Multi-class: cars, pedestrians, animals or weapon
- Cloud based or device based

**Non-functional**

- Target mAP > 0.75 at IOU 0.5
- Latency < 100ms per frame for real-time applications
- Scalable to handle multiple concurrent requests
- High availability (99.9%)

## 2. ML task & pipeline

```text
Input Stream -> Preprocessor -> Detection Model -> Post-processor -> Output
     ↑                             ↑                    ↑
     |                             |                    |
Data Pipeline                Model Registry         NMS/Filtering
```

## 3. data

**Data Collection**

- Public datasets: COCO, Pascal VOC, OpenImages
- Custom collected data for specific use cases
- Synthetic data generation for rare cases (especially weapons)

**Data Storage**

- Raw images: MinIO object storage
- Annotations: MongoDB (flexible schema for different annotation formats)
- Features: Vector database (FAISS/Milvus)

**Data Pipeline**

- Preprocessing
  - Resize: 640x640 or 1024x1024 based on model
  - Normalization: mean subtraction, scaling to [0,1]
  - Augmentation
    - Geometric: rotation, flip, scale
    - Photometric: brightness, contrast, noise
    - Mosaic augmentation for small objects
    - Mixup for regularization
- Annotation
  - Bounding box format: [x_center, y_center, width, height]
  - Class labels
  - Quality checks: IOU overlaps, size constraints

## 4. model

**Cloud Deployment:**

- Primary: YOLOv7 or YOLOv8
  - Better accuracy-speed trade-off
  - Strong multi-scale detection
  - Built-in data augmentation

**Edge Deployment:**

- Primary: YOLOv8-nano or SSD-MobileNetV3
  - Optimized for mobile/edge
  - Reduced parameter count
  - TensorRT/ONNX compatible

### two-stage: region proposal and object classification

- generates a set of potential object bounding boxes
- takes the proposed regions from the RPN and classifies them into different object categories

### one-stage

- perform both region proposal and object classification in a single step

### nms

## 5. evaluation

**Primary Metrics:**

- mAP@0.5: Overall detection performance
- mAP@0.5:0.95: Stricter evaluation
  - AP: avg. across various IOU thresholds
- FPS: Runtime performance
- Per-class AP for monitoring class-wise performance

**Secondary Metrics:**

- Precision-Recall curves
  - Precision based on IOU threshold
- F1 score at different confidence thresholds
- Average inference time

## 6. deploy & serving

- batch service or online service

## 7. Monitoring & maintenance

**monitoring**

- Data drift detection
- A/B testing framework
- Regular model retraining pipeline
- Performance optimization based on real-world feedback

## 8. 问答

- 过杀和漏检：基于遗传算法的帕累托优化

## Reference

- [仪表识别](https://github.com/hjptriplebee/meterReader)
- [安全帽](https://github.com/PeterH0323/Smart_Construction)
- [万字长文细说工业缺陷检测 - 皮特潘的文章 - 知乎](https://zhuanlan.zhihu.com/p/375828501)
- [https://github.com/Charmve/Surface-Defect-Detection](https://github.com/Charmve/Surface-Defect-Detection)
- [https://github.com/Sharpiless/Yolov5-Flask-VUE](https://github.com/Sharpiless/Yolov5-Flask-VUE)
- [Meta (Facebook) Machine Learning Case Study: Illegal Items Detection](https://jayfeng.medium.com/meta-facebook-machine-learning-case-study-illegal-items-detection-b5e5a4e8afd0)
- [通用目标检测开源框架YOLOv6在美团的量化部署实战](https://tech.meituan.com/2022/09/22/yolov6-quantization-in-meituan.html)
- [基于多模态信息抽取的菜品知识图谱构建](https://mp.weixin.qq.com/s/0isxFC4iVrMuNseFil7xRQ)
- [美团视觉GPU推理服务部署架构优化实践](https://tech.meituan.com/2023/02/09/inference-optimization-on-gpu-by-meituan-vision.html)
