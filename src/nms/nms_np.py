# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# 说明来自fast rcnn源码，基本都是，太经典了呀

import numpy as np

# dets:(nums, 5[x1, y1, y1, y2])
def py_cpu_nms(dets, thresh):     
    # 获取bbox的对角坐标，用于计算iou
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] 
    y2 = dets[:, 3]
    score = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算boxes面积 + 防止出现0
    # det值不动，通过order来维护
    order = score.argsort()[::-1]    # 对pred bbox按score做降序排序，对应step-2
    keep = [] # NMS后最终的pred bbox
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        over = inter / (areas[i] + areas[order[1:]] - inter)
        # over <= thresh的非冗余bbox，其inds保留下来，作进一步筛选
        inds = np.where(over <= thresh)[0]
       
        # 保留有效bbox，就是这轮NMS未被抑制掉的幸运儿，为什么 + 1？
        # 因为ind = 0就是这轮NMS的top-1，剩余有效bbox在IoU计算中与top-1做的计算，inds对应回原数组，自然要做 +1 的映射，接下来就是step-4的循环
        order = order[inds + 1]
    return keep

if __name__ == '__main__':
    dets = np.array([[100,120,170,200,0.98],
                [20,40,80,90,0.99],
                [20,38,82,88,0.96],
                [200,380,282,488,0.9],
                [19,38,75,91, 0.8]])

    keep = py_cpu_nms(dets, 0.5)
    print(keep)

    