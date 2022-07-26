import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os

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
    print("name: %s, shape: %s" % (str(path), str(img.shape)))

def get_pixel_value(path, type="RGB"):
    img = Image.open(path)
    width = img.size[0]
    height = img.size[1]
    for h in range(0, height):
        for w in range(0, width):
            pixel = img.getpixel((w, h))
            if type == "RGB":
                for i in range(0, 3):
                    print(" %d " % pixel[i])
            else:
                print(" %d " % pixel) # 0和1

"""
（只将颜色从黑红改成黑白，通道数没变，现在没啥用）
"""
def change_labels(from_dir=r"D:\beam-transfer\train\labels",
                  to_dir=r"D:\beam-transfer\train\labels_2",
                  ids = []):
    if len(ids) > 0:
        for id in ids:
            file = str(id) + ".png"
            img = cv2.imread(os.path.join(from_dir, file))
            b, g, r = cv2.split(img)
            r[np.where(r != 0)] = 255
            cv2.imwrite(os.path.join(to_dir, file), r)
    else:
        for file in os.listdir(from_dir):
            img = cv2.imread(os.path.join(from_dir, file))
            b, g, r = cv2.split(img)
            r[np.where(r != 0)] = 255
            cv2.imwrite(os.path.join(to_dir, file), r)




if __name__ == "__main__":
    get_pic_shape(r"D:\beam-transfer\train\labels\100.png")
    # get_pic_shape(r"D:\beam-transfer\try\100.png")
    # get_pixel_value(r"D:\beam-transfer\train\labels\100.png", type="single")
    change_labels(ids=[100])
    get_pic_shape(r"D:\beam-transfer\train\labels_2\100.png")
