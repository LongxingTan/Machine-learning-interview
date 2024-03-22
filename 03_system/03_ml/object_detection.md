# 2D目标检测

## 1. requirements
- Cloud based or device based deployment


## 2. metrics
- Precision
- calculated based on IOU threshold
- AP: avg. across various IOU thresholds
- mAP: mean of AP over C classes

## 3. data collection


## 4. feature


## 5. model

### 2-stage
- generates a set of potential object bounding boxes
- takes the proposed regions from the RPN and classifies them into different object categories


### 1-stage
- perform both region proposal and object classification in a single step
