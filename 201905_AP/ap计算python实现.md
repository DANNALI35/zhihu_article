
AP概念理解可参考上篇[【目标检测】VOC mAP](https://zhuanlan.zhihu.com/p/67279824)。本篇着重解析AP计算python代码。

## Imports


```python
import numpy as np
```

## bbox 排序

bbox判断为FP/FP的标准有2个：bbox属于某类别得分、bbox与gt的iou，**在满足iou阈值的条件下，bbox得分高优先标记为TP**。

例如：对于某gt，其对应2个预测的bbox a和b，bbox a的类别得分是0.8，与该gt的iou是0.6，bbox b的类别得分是0.6，与该gt的iou是0.7，此时应标记bbox a为tp，bbox b为fp。

因此，AP计算的第一步，我们先针对每个图片样本，将其预测的bbox（经nms之后的）按类别得分由高到低排列。

### 函数


```python
def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
    pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores
```

### 例子


```python
pred_boxes = [np.array([[1, 2, 3, 12, 32, 48], [1, 9, 9, 12, 32, 43], [1, 2, 6, 22, 42, 11]]),
              np.array([[22, 22, 23, 42, 42, 63], [1, 2, 3, 22, 42, 13], [1, 2, 3, 22, 42, 14]])]
pred_labels = [np.array([1, 1, 2]), np.array([1, 3, 2])]
pred_scores = [np.array([0.7,0.8, 0.3]), np.array([0.6, 0.7, 0.2])]
pred_boxes, pred_labels, pred_scores = sort_by_score(pred_boxes, pred_labels, pred_scores)
print(pred_boxes, '\n',pred_labels, '\n', pred_scores)
```

    '''out
    [array([[ 1,  9,  9, 12, 32, 43],
           [ 1,  2,  3, 12, 32, 48],
           [ 1,  2,  6, 22, 42, 11]]), 
     array([[ 1,  2,  3, 22, 42, 13],
           [22, 22, 23, 42, 42, 63],
           [ 1,  2,  3, 22, 42, 14]])] 
     [array([1, 1, 2]), array([3, 1, 2])] 
     [array([0.8, 0.7, 0.3]), array([0.7, 0.6, 0.2])]
     '''

## iou计算
以3d box和gt为例，其iou计算过程如下：

### 函数


```python
def iou_3d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(y1,x1,z1,y2,x2,z2)]
    :param cubes_b: [M,(y1,x1,z1,y2,x2,z2)]
    :return:  IoU [N,M]
    """
    # 扩维
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]

    # 分别计算高度和宽度的交集
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 3:], cubes_b[..., 3:]) -
                         np.maximum(cubes_a[..., :3], cubes_b[..., :3]))  # [N,M,(h,w,t)]

    # 交集
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # 计算面积
    area_a = np.prod(cubes_a[..., 3:] - cubes_a[..., :3], axis=-1)
    area_b = np.prod(cubes_b[..., 3:] - cubes_b[..., :3], axis=-1)

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou

```

### 例子


```python
gt_boxes = np.array([[1, 2, 3, 12, 32, 43], [1, 2, 6, 22, 42, 10]])
pred_boxes = np.array([[1, 2, 3, 12, 32, 48], [1, 9, 9, 12, 32, 43], [1, 2, 6, 22, 42, 11]])
iou_3d(gt_boxes,pred_boxes )
```


    '''out
    array([[0.88888889, 0.65166667, 0.1047619 ],
           [0.07815275, 0.02160731, 0.8       ]])
    '''

## 根据PR曲线计算AP
以下函数来自[faster rcnn工程](https://github.com/rbgirshick/py-faster-rcnn),写得相当巧妙，通过所有点插值法计算AP。

### 函数


```python
def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
```

### 例子


```python
recall = np.array([0.3,0.4,0.6]) 
precision = np.array([1.,0.5,0.3])
_compute_ap(recall, precision)
```


    '''out
    0.41
    '''

## AP计算主逻辑

* 先针对每个图片样本，将其预测的bbox（经nms之后的）按类别得分由高到低排列
* 循环存在类别
    * 获取某一类对应的样本的gt及其预测的box和score
    * 对该类样本循环
        * 对样本预测的box循环
            * 如果box预测的label在样本中不存在，直接将该box标记为fp
            * 根据与gt的iou判断box是tp还是fp，并保证1个gt只有一个tp与之对应
            * 将tp、fp、gt的score append到list中
        * 将score list由大至小排序，tp list和fp list也相应排序
        * 将排序后的tp list和fp list进行累加，并计算得到precision list 和recall list
        * 通过precision list 和recall list（即：PR曲线）计算得AP

### 函数


```python
def eval_ap_3d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(y1,x1,z1,y2,x2,z2)),(b,(y1,x1,z1,y2,x2,z2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(y1,x1,z1,y2,x2,z2)),(n,(y1,x1,z1,y2,x2,z2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 3, 0 for background,only calculate cls index in range(num_cls)[1:]
    :return: a dict containing average precision for each cls
    """
    pred_boxes, pred_labels, pred_scores = sort_by_score(pred_boxes, pred_labels, pred_scores)
    all_ap = {}
    for label in range(num_cls)[1:]:
        # get samples with specific label
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]

        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        fp = np.zeros((0,))
        tp = np.zeros((0,))
        scores = np.zeros((0,))
        total_gts = 0
        # loop for each sample
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            for index in range(len(sample_pred_box)):
                scores = np.append(scores, sample_scores[index])
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                iou = iou_3d(sample_gts, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores) 
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        print(recall, precision)
        ap = _compute_ap(recall, precision)
        all_ap[label] = ap
    return all_ap
```

### 例子


```python
iou_thread = 0.5
gt_boxes = [np.array([[1, 2, 3, 12, 32, 43], [1, 2, 3, 12, 32, 42], [1, 2, 6, 22, 42, 10]]),
            np.array([[13, 2, 3, 16, 32, 43]])]
gt_labels = [np.array([1, 1, 2]), np.array([1])]

pred_boxes = [np.array([[1, 2, 3, 12, 32, 48], [1, 9, 9, 12, 32, 43], [1, 2, 6, 22, 42, 11]]),
              np.array([[22, 22, 23, 42, 42, 63], [1, 2, 3, 22, 42, 13], [1, 2, 3, 22, 42, 14]])]
pred_labels = [np.array([1, 1, 2]), np.array([1, 3, 2])]
pred_scores = [np.array([0.7, 0.8, 0.3]), np.array([0.6, 0.7, 0.2])]
num_cls = 4
pred_boxes, pred_labels, pred_scores = sort_by_score(pred_boxes, pred_labels, pred_scores)
all_ap = eval_ap_3d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls)
print('AP for all cls:', all_ap)
```

    '''out
    [0.33333333 0.33333333 0.33333333] [1.         0.5        0.33333333]
    [1. 1.] [1.  0.5]
    [nan] [0.]
    AP for all cls: {1: 0.3333333333333333, 2: 1.0, 3: nan}
    '''

## ALL in One

```python
# -*- coding: utf-8 -*- 
"""
@project: 201904_dual_shot
@file: AP.py
@author: danna.li
@time: 2019-06-11 18:20
@description: 
"""

import numpy as np


def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
    pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores


def iou_3d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(y1,x1,z1,y2,x2,z2)]
    :param cubes_b: [M,(y1,x1,z1,y2,x2,z2)]
    :return:  IoU [N,M]
    """
    # 扩维
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]

    # 分别计算高度和宽度的交集
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 3:], cubes_b[..., 3:]) -
                         np.maximum(cubes_a[..., :3], cubes_b[..., :3]))  # [N,M,(h,w,t)]

    # 交集
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # 计算面积
    area_a = np.prod(cubes_a[..., 3:] - cubes_a[..., :3], axis=-1)
    area_b = np.prod(cubes_b[..., 3:] - cubes_b[..., :3], axis=-1)

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_ap_3d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(y1,x1,z1,y2,x2,z2)),(b,(y1,x1,z1,y2,x2,z2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(y1,x1,z1,y2,x2,z2)),(n,(y1,x1,z1,y2,x2,z2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 4, total number of class including background which is equal to 0
    :return: a dict containing average precision for each cls
    """
    all_ap = {}
    for label in range(num_cls)[1:]:
        # get samples with specific label
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]

        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        fp = np.zeros((0,))
        tp = np.zeros((0,))
        scores = np.zeros((0,))
        total_gts = 0
        # loop for each sample
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            for index in range(len(sample_pred_box)):
                scores = np.append(scores, sample_scores[index])
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                iou = iou_3d(sample_gts, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _compute_ap(recall, precision)
        all_ap[label] = ap
        print(recall, precision)
    return all_ap


def test():
    iou_thread = 0.5
    gt_boxes = [np.array([[1, 2, 3, 12, 32, 43], [1, 2, 3, 12, 32, 42], [1, 2, 6, 22, 42, 10]]),
                np.array([[13, 2, 3, 16, 32, 43]])]
    gt_labels = [np.array([1, 1, 2]), np.array([1])]

    pred_boxes = [np.array([[1, 2, 3, 12, 32, 48], [1, 9, 9, 12, 32, 43], [1, 2, 6, 22, 42, 11]]),
                  np.array([[22, 22, 23, 42, 42, 63], [1, 2, 3, 22, 42, 13], [1, 2, 3, 22, 42, 14]])]
    pred_labels = [np.array([1, 1, 2]), np.array([1, 3, 2])]
    pred_scores = [np.array([0.7, 0.8, 0.3]), np.array([0.6, 0.7, 0.2])]
    num_cls = 4
    pred_boxes, pred_labels, pred_scores = sort_by_score(pred_boxes, pred_labels, pred_scores)
    all_ap = eval_ap_3d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls)
    print('AP for all cls:', all_ap)


if __name__ == '__main__':
    test()
```

    [0.33333333 0.33333333 0.33333333] [1.         0.5        0.33333333]
    [1. 1.] [1.  0.5]
    [nan] [0.]
    AP for all cls: {1: 0.3333333333333333, 2: 1.0, 3: nan}

