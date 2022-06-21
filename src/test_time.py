import numpy as np 
import time

from nms.nms_np import py_cpu_nms
# from nms.nms_np_c import py_cpu_nms
# 生成6000个boxes
np.random.seed( 1 )   # keep fixed
num_rois = 6000
minxy = np.random.randint(50,145,size=(num_rois ,2))
maxxy = np.random.randint(150,200,size=(num_rois ,2))
score = 0.8*np.random.random_sample((num_rois ,1))+0.2
 
boxes_new = np.concatenate((minxy,maxxy,score), axis=1).astype(np.float32)

def nms_test_time(boxes_new):
     
    thresh = [0.7,0.8,0.9]
    T = 50   # 重复次数
    for i in range(len(thresh)):
        since = time.time()
        for t in range(T):
            keep = py_cpu_nms(boxes_new, thresh=thresh[i])     # for cpu
#            keep = gpu_nms(boxes_new, thresh=thresh[i])       # for gpu
        print("thresh={:.1f}, time wastes:{:.4f}".format(thresh[i], (time.time()-since)/T))
     
    return keep

if __name__ =="__main__":
    nms_test_time(boxes_new)