# 2D目标检测

## 1. requirements
- Cloud based or device based deployment


## 2. data collection


## 3. feature


## 4. model

### 2-stage: region proposal and object classification
- generates a set of potential object bounding boxes
- takes the proposed regions from the RPN and classifies them into different object categories


### 1-stage
- perform both region proposal and object classification in a single step


### nms


## 5. evaluation
- Precision
- calculated based on IOU threshold
- AP: avg. across various IOU thresholds
- mAP: mean of AP over C classes

## 6. deploy & serving


## 问答
- 过杀和漏检：基于遗传算法的帕累托优化


## Reference
- [仪表识别](https://github.com/hjptriplebee/meterReader)
- [安全帽](https://github.com/PeterH0323/Smart_Construction)
- [万字长文细说工业缺陷检测 - 皮特潘的文章 - 知乎](https://zhuanlan.zhihu.com/p/375828501)
- [https://github.com/Charmve/Surface-Defect-Detection](https://github.com/Charmve/Surface-Defect-Detection)
- [https://github.com/Sharpiless/Yolov5-Flask-VUE](https://github.com/Sharpiless/Yolov5-Flask-VUE)
- [Meta (Facebook) Machine Learning Case Study: Illegal Items Detection](https://jayfeng.medium.com/meta-facebook-machine-learning-case-study-illegal-items-detection-b5e5a4e8afd0)
