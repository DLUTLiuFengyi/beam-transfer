import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

"""
基础款模型
"""
class SimpleNet(nn.Module):
    def __init__(self, out_ch=2):
        super(SimpleNet, self).__init__()
        # 下面3个卷积层
        self.conv1 = nn.Sequential(
            # Conv2d函数参数：输入通道数, 输出通道数, kernel_size卷积核大小, padding边界0填充, dilation卷积核之间间距
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
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
        # 如果是语义分割任务，则这里继续用卷积
        self.lb = nn.Sequential(
            nn.Conv2d(64, 2, 3, 1, 1)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        lb = self.lb(conv3_out) # shape: batch_size * 2 * 60 * 80
        # interpolate函数用来对输入的特征图做插值放大，用双线性插值法
        out = F.interpolate(lb, x.size()[2:], mode="bilinear", align_corners=True) #
        return out # shape: batch_size * 2 * 480 * 640


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()

        # 两层卷积加激活
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1),  # padding=1，希望图像经过卷积之后大小不变。
            nn.ReLU(inplace=True),  # inplace=True，就地改变

            # 第二层卷积，输入是多少个channel，输出仍然是多少个channel。
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 下采样
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, is_pool=True):
        if is_pool:  # 是否需要进行下采样
            x = self.pool(x)  # 先下采样，再卷积。
        x = self.conv_relu(x)
        return x


# 上采样模型。卷积、卷积、上采样（反卷积实现上采样）
class Upsample(nn.Module):
    # 上采样中，输出channel会变成输入channel的一半。所以并不需要给两个channel值。看unet网络模型就知道了。
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(2 * channels, channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第二层卷积，channel不变。
            nn.Conv2d(channels, channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 上采样，也需要激活。
        self.upconv_relu = nn.Sequential(
            # 反卷积层
            nn.ConvTranspose2d(channels,  # 输入channel
                               channels // 2,  # 输出channel
                               kernel_size=3,
                               stride=2,  # 跨度必须为2，才能放大，长和宽会变为原来的两倍
                               padding=1,  # 输入的kernel所在的位置，起始位置在图像的第一个像素做反卷积。
                               output_padding=1),  # 反卷积之后，在最外层（周边）做的填充。边缘填充为1，
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_relu(x)
        x = self.upconv_relu(x)
        return x


# 创建Unet。我们要初始化上、下采样层，还有其他的一些层
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = Downsample(3, 64)  # 输入的是一张图片，channel是3。
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 1024)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(1024,
                               512,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = Upsample(512)
        self.up2 = Upsample(256)
        self.up3 = Upsample(128)

        self.conv_2 = Downsample(128, 64)  # 最后的两层卷积
        self.last = nn.Conv2d(64, 2, kernel_size=1)  # 输出层。

    # 前向传播。
    def forward(self, x):
        x1 = self.down1(x, is_pool=False)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x5 = self.up(x5)

        # 我们需要将x4的输出将x5（up上采样的输出）做一个合并。使得channel变厚。
        # 将前面下采样的特征合并过来,有利于重复利用这些特征，有利于模型的每一个像素进行分类。

        x5 = torch.cat([x4, x5], dim=1)  # 32*32*1024。沿着channel这个维度进行合并。[batch channel height weight]
        x5 = self.up1(x5)  # 64*64*256) # 上采样
        x5 = torch.cat([x3, x5], dim=1)  # 64*64*512
        x5 = self.up2(x5)  # 128*128*128
        x5 = torch.cat([x2, x5], dim=1)  # 128*128*256
        x5 = self.up3(x5)  # 256*256*64
        x5 = torch.cat([x1, x5], dim=1)  # 256*256*128

        x5 = self.conv_2(x5, is_pool=False)  # 256*256*64。最后的两层卷积

        x5 = self.last(x5)  # 256*256*3 # 最后的输出层。每一个像素点都输出为长度为2的向量。
        return x5

if __name__ == "__main__":
    # 新建一个随机图，测试
    img = torch.rand(1, 3, 480, 640)
    model = SimpleNet()
    # model = UNet()
    out = model(img)
    print(out.shape) # torch.Size([1, 2, 480, 640])