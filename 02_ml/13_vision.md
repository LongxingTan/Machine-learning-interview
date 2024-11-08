# 机器视觉


## 1. 模型
- ConvNext
- ViT

### 分类


### 检测
参考[物体检测系统设计](../03_system/03_ml/object_detection.md)

- 二阶段检测
- 一阶段检测
  - YOLO
- 旋转目标检测


### 分割


### 视频


## 2. 代码

**IOU**
```python

def iou(box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # 左上角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2),   # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),   # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0   # 两个box没有重叠区域
    inter = wh[:,:,0] * wh[:,:,1]   # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # (N,)
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N,M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N,M)  # (N,M)

    iou = inter / (area1+area2-inter)
    return iou
```

```python
import numpy as np

def compute_iou(boxA, boxB):
    # box:(x1,y1,x2,y2), x1,y1为左上角。原点为左上角，x朝右为正，y朝下为正。
    # 计算相交框的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交区域，并区域，及IOU。要和0比较大小，如果是负数就说明压根不相交
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

boxA = [1,1,3,3]
boxB = [2,2,4,4]
IOU = ComputeIOU(boxA, boxB)
```


- nms
```python
def nms(boxes, scores, threshold):
    # boxes: 边界框列表，每个框是一个格式为 [x1, y1, x2, y2] 的列表
    # scores: 每个边界框的得分列表
    # threshold: NMS的IoU阈值

    # 按得分升序排列边界框
    sorted_indices = np.argsort(scores)
    boxes = [boxes[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    keep = []  # 保留的边界框的索引列表

    while boxes:
        # 取得分最高的边界框
        current_box = boxes.pop()
        current_score = scores.pop()

        keep.append(sorted_indices[-1])
        sorted_indices = sorted_indices[:-1]
        discard_indices = []  # 需要丢弃的边界框的索引列表

        for i, box in enumerate(boxes):
            # 计算与当前边界框的IoU
            iou = compute_iou(current_box, box)

            # 如果IoU超过阈值，标记该边界框为需要丢弃
            if iou > threshold:
                discard_indices.append(i)

        # 移除标记为需要丢弃的边界框。从后往前删，不然for循环会出错
        for i in sorted(discard_indices, reverse=True):
            boxes.pop(i)
            scores.pop(i)
            sorted_indices = np.delete(sorted_indices, i) # np与list的方法不同

    return keep
```

## 参考
- [Randaugment: Practical automated data augmentation with a reduced search space](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf)
-
