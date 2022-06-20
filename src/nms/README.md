## 为什么需要NMS？
简言之，过滤掉重复的预测框
由于为保证对目标的高召回，就必须使用滑窗、anchor / default bbox密集采样的方式，这就导致同一目标会有多个匹配的锚框。
最简单就就是设置较低的cls score 去过滤，但这样并不完全，那过滤不干净的就交给NMS去实现吧。

## NMS实现步骤
step1: 将所有bbox按照类别进行划分(如pascal voc分20个类，也即将output_bbox按照其对应的cls score划分为21个集合，1个bg类，只不过bg类就没必要做NMS而已)
step2: 在每一个集合内，根据每个bbox的cls score进行降序排列， 达到list_k
step3: 从list_k中的top1 开始（选个老大），然后计算与其他bbox（小弟）的IOU，如果IOU大于阈值T(常为0.5)则剔除出list_k(说明和老大太像了，一山不容二虎)，得到经过一轮剔除后的list_k
step4: 选择list_k中top2-最后一个，重复步骤三，直到所有list_k都完成筛选

**小疑惑**
- 所哟list_k都剔除一遍了，就能刚好保证一个目标就一个bbox吗，不会出现刚好卡在IOU阈值内都保留下来的情况吗？

