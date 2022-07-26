import os.path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
        # 用PIL打开掩码图,如果用cv2打开的话还需单独转换成灰度图
        mask = Image.open(label_name)
        # resize，同时用最近邻算法插值
        mask = mask.resize((self.width, self.height), Image.NEAREST)
        # target是最终训练用的标签 png -> numpy的数组 -> tensor
        target = np.array(mask).astype('int32') # shape: H * W

        return img, torch.from_numpy(target).long()

    """
    需要重写len函数，否则调用torch的dataloader.py时报错
    """
    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    train_data = BeamDataset(height=480, width=640, name_file="train.txt")
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

    img_batch, img_lab_batch = next(iter(train_loader))
    print(img_batch.shape)  # shape: batch_size * H * W * 3
    print(img_lab_batch.shape) # shape: batch_size * H * W

    img = img_batch[0].numpy()
    img_lab = img_lab_batch[0].numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)

    plt.imshow(img_lab)
    plt.show()
