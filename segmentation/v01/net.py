import torch
import torch.nn as nn
import torch.nn.functional as F

"""
基础款模型
"""
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 下面3个卷积层
        self.conv1 = nn.Sequential(
            # Conv2d函数参数：输入通道数, 输出通道数, kernel_size卷积核大小, padding边界0填充, dilation卷积核之间间距
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) # 32*14*14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 如果是图像分类任务，则这里是全连接层
        # 如果是语义分割任务，则这里继续用卷积，最后输出图片而不是数组
        self.lb = nn.Sequential(
            nn.Conv2d(64, 2, 3, 1, 1)
        )

    def forward(self, x):
        # x.size()  batch_size * 3 * H * W
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        lb = self.lb(conv3_out)
        # interpolate函数用来对输入的特征图做插值放大，用双线性插值法
        out = F.interpolate(lb, x.size()[2:], mode="bilinear", align_corners=True)
        # out.size()  batch_size * 2 * H * W
        return out

