# 2D目标检测

## 1. requirements
- Cloud based or device based deployment


## 2. ML task & pipeline


## 3. data collection


## 4. model

### two-stage: region proposal and object classification
- generates a set of potential object bounding boxes
- takes the proposed regions from the RPN and classifies them into different object categories


### one-stage
- perform both region proposal and object classification in a single step


### nms


## 5. evaluation
- Precision based on IOU threshold
- AP: avg. across various IOU thresholds
- mAP: mean of AP over C classes

## 6. deploy & serving


## 7. 问答
- 过杀和漏检：基于遗传算法的帕累托优化


## Reference
- [仪表识别](https://github.com/hjptriplebee/meterReader)
- [安全帽](https://github.com/PeterH0323/Smart_Construction)
- [万字长文细说工业缺陷检测 - 皮特潘的文章 - 知乎](https://zhuanlan.zhihu.com/p/375828501)
- [https://github.com/Charmve/Surface-Defect-Detection](https://github.com/Charmve/Surface-Defect-Detection)
- [https://github.com/Sharpiless/Yolov5-Flask-VUE](https://github.com/Sharpiless/Yolov5-Flask-VUE)
- [Meta (Facebook) Machine Learning Case Study: Illegal Items Detection](https://jayfeng.medium.com/meta-facebook-machine-learning-case-study-illegal-items-detection-b5e5a4e8afd0)
- [基于多模态信息抽取的菜品知识图谱构建](https://mp.weixin.qq.com/s/0isxFC4iVrMuNseFil7xRQ)
