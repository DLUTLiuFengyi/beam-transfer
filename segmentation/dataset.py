import os.path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class BeamDataset(Dataset):
    def __init__(self, root=r"D:\beam-transfer\train", name_file="train.txt", train_img_dir="beams", train_label_dir="labels",
                 height=480, width=640,
                 transform=None, target_transform=None):
        txt = open(os.path.join(root, name_file), 'r')
        imgs = []
        for line in txt:
            line = line.strip('\n') # 删去换行符
            # 原始图 + 标签图 构成的元组
            # print(os.path.join(root, train_img_dir, line + ".png"))
            imgs.append((os.path.join(root, train_img_dir, line + ".png"), os.path.join(root, train_label_dir, line + ".png")))
        self.height = height
        self.width = width
        self.imgs = imgs
        # transform用来对原始图进行变换，包括归一化、转变成tensor等操作
        self.transform = transform

    """
    获得训练数据中第index个图像和它对应的标签
    """
    def __getitem__(self, index):
        # 图像名，标签名
        img_name, label_name = self.imgs[index]
        # 加载彩色图片
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        # resize，使所有图片的尺寸保持一致
        img = cv2.resize(img, (self.width, self.height))
        if self.transform is not None:
            img = self.transform(img)
        # 掩码图
        # 用PIL打开掩码图好处是PIL能直接这样打开png文件（否则用cv2打开的话还需单独转换成灰度图）
        mask = Image.open(label_name)
        # resize，同时用最近邻算法插值
        mask = mask.resize((self.width, self.height), Image.NEAREST)
        # target是最终训练用的标签 png -> numpy的数组 -> tensor
        target = np.array(mask).astype('int32')
        # 产生一个跟原来的target数组相同大小的值为true或false的数组，令原target数组中值为255的变成-1
        target[target == 255] = -1 # 没有这步的话，训练好后的模型的预测结果是空白
        return img, torch.from_numpy(target).long()

    """
    需要重写len函数，否则调用torch的dataloader.py时报错
    """
    def __len__(self):
        return len(self.imgs)