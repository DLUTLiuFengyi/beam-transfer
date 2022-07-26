import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

"""
判断是否能用gpu
"""
def judge_cuda():
    if torch.cuda.is_available():
        return True
    else:
        return False

"""
获取图片形状
"""
def get_pic_shape(path):
    img = Image.open(path)
    img = np.array(img)
    print(img.shape)

"""
cv2方式打开图片
"""
def get_tensor_from_img_by_cv2(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 480))
    img_transfer = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = img_transfer(img).unsqueeze(0) # unsqueeze(0)作用：3*480*640 -> 1*3*480*640
    return img_tensor

"""
PIL方式打开图片
"""
def get_tensor_from_img_by_PIL(img_path):
    img = Image.open(img_path)
    img_transfer = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = img_transfer(img).unsqueeze(0)[:, 0:3, :, :]
    return img_tensor

"""
对numpy数组进行一系列处理：归一化 -> 乘255 -> 目标区域设置成白色，非目标区域设置成黑色
这里的array是二维数组
目的是让numpy数组转换成的图片符合下一步蒙版合成的要求
如果不归一化，array数组的值基本是1到5之间，放到[0, 255]的画板上后，画出来基本是黑色
"""
def normalize_array(array):
    [rows, cols] = array.shape
    max = -1.0
    min = 256.0
    # 更新max与min的值
    for i in range(rows):
        for j in range(cols):
            if array[i, j] > max:
                max = array[i, j]
            if array[i, j] < min:
                min = array[i, j]
    print("max = %f, min = %f" % (max, min))
    # 归一化 -> 乘255 -> 目标区域设置成白色，非目标区域设置成黑色
    for i in range(rows):
        for j in range(cols):
            array[i, j] = ((array[i, j] - min) / (max-min)) * 255
            if array[i, j] < 30:
                array[i, j] = 255
            else:
                array[i, j] = 0
    return array
