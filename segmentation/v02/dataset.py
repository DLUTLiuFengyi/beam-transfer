import os.path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

"""
root: 训练集和验证集安排在同一目录下
name_file: 记录训练集或验证集所用图片文件名的txt
"""
class BeamDataset(Dataset):
    def __init__(self, root=r"D:\beam-transfer\train", name_file="train.txt",
                 train_img_dir="beams", train_label_dir="labels",
                 height=480, width=640,
                 transform_compos=None):
        txt = open(os.path.join(root, name_file), 'r')
        imgs = []
        for line in txt:
            line = line.strip('\n') # 删去换行符
            # 原始图路径 + 标签图路径 构成的元组
            imgs.append((os.path.join(root, train_img_dir, line + ".png"),
                         os.path.join(root, train_label_dir, line + ".png")))
        self.height = height
        self.width = width
        self.imgs = imgs
        # 定义对原始图进行转换的转换器，包括调整尺寸、图片对象转成tensor对象、归一化等操作
        if transform_compos is not None:
            self.transform_compos = transform_compos
        else:
            self.transform_compos = transforms.Compose([
                transforms.Resize([self.height, self.width]),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # 获得训练数据中第index个图像和它对应的标签图
    def __getitem__(self, index):
        # 原图路径，标签图路径
        img_name, label_name = self.imgs[index]
        # 读取原图
        img = Image.open(img_name).convert("RGB")
        # 读取标签图
        img_lab = Image.open(label_name)

        # 对原图作转换
        img = self.transform_compos(img)
        # 对标签图作转换
        img_lab = img_lab.resize((self.width, self.height), Image.NEAREST)
        img_lab = np.array(img_lab) # (480, 640)
        img_lab = torch.from_numpy(img_lab).long() # torch.Size([480, 640])
        # # 本场景的语义分割是二分类问题，标签图的tensor取值只能为0或1
        img_lab[img_lab > 0] = 1

        return img, img_lab

    # 需要重写len函数，否则调用torch的dataloader时报错
    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    train_data = BeamDataset(height=256, width=256, name_file="train.txt")
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    # eval_data = BeamDataset(height=480, width=640, name_file="eval.txt")
    # eval_loader = DataLoader(dataset=eval_data, batch_size=8)

    img_batch, img_lab_batch = next(iter(train_loader))
    print(img_batch.shape)  # shape: batch_size * 3 * H * W
    print(img_lab_batch.shape) # shape: batch_size * H * W

    img = img_batch[0].permute(1, 2, 0).numpy()  # permute方法调整各维度位置，让plt正常画图
                                                 # 1*3*480*640 -> 1*480*640*3
    img_lab = img_lab_batch[0].numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)

    plt.imshow(img_lab)
    plt.show()
