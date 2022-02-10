'''
2022/1/18
目的：分被用opencv 和 PIL导入图像并转为标准图像tensor格式，最后再还原显示图片
'''
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
# 通过opencv导入图像 类型：numpy的ndarray （H，W，C=3），通道顺序（B,G,R)
img1 = cv2.imread('D:\\Desktop\\my_CVcode\\img\\test1.jpg')
print('img1 size:{}'.format(img1.shape))     
# 通过image读入图像 类型：JpegImageFile   (W，H)  
img2 = Image.open('D:\\Desktop\\my_CVcode\\img\\test1.jpg')  
# JpegImageFile 转为ndarray
# img2 = np.asarray(img2)
print('img2 size:{}'.format(img2.size))  
img2.show()
# 转为标准图像tensor（和torch.from_numpy()有区别)（H，W，C=3） 通道顺序（R,G,B)
img1_tensor = transforms.ToTensor()(img1)
img2_tensor = transforms.ToTensor()(img2)
print('img1_tensor size:{}'.format(img1_tensor.shape))  
print('img2_tensor size:{}'.format(img2_tensor.shape))  

# 将图像tensor还原为PIL格式(torvision专门的函数)
img_pil = transforms.ToPILImage()(img1_tensor)

# 利用Image的show类显示图片
#TODO: 存在bug,显示颜色不对
img_pil.show()

#TODO:用opencv显示还原的图像tensor
