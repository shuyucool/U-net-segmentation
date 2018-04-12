import matplotlib.pyplot as plt
import numpy as np
from data import *
from unet import *
import cv2

"""此程序是为了将原图与其预测图一一对应进行保存，因为在读取测试图像时并不是按照图像序号顺序读取的。"""

path_Con_test_image = "data/Con_test_image1" # 原图输出文件路径
path_Con_test_lable = "data/Con_test_lable1" # 预测图像输出文件路径

imgs_test = np.load('data/npydata/imgs_test.npy')

imgs_test_predict = np.load('imgs_mask_test_1.npy')
imgs_test_predict *= 255
# TODO：这里之所以乘以255，在转换为“uint8”类型，是因为“float32”无法保存成“.tif”图片
imgs_test_predict = imgs_test_predict.astype('uint8')
print(imgs_test.shape, imgs_test_predict.shape)

for i in range(0,222):
    cv2.imwrite(path_Con_test_image + "/" + str(i) + "." + "tif", imgs_test[i])  # 保存测试图像
    cv2.imwrite(path_Con_test_lable + "/" + str(i) + "." + "tif", imgs_test_predict[i])  # 保存测试图像的label