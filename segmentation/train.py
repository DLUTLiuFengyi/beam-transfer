import torch
import math
from torchvision import transforms
from dataset import BeamDataset
import segmentation.net as snet
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from beam_utils import judge_cuda

"""
训练模型
"""
def train(pth_dir=r"D:\beam-transfer\pth"):
    # os.makedirs(pth_dir, exist_ok=True)
    height = 600
    width = 800
    batch_size=8
    epochs = 10
    learning_rate = 0.04
    # height与width太大的话，GPU显存不够用
    train_data = BeamDataset(height=height, width=width, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    model = snet.SimpleNet()
    if judge_cuda():
        model.cuda()

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    # BeamDataset的__getitem__方法中target有的取值会为-1，这里不ignore的话会报错
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            # 获取训练所用的真实图x和标签图y
            if judge_cuda():
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            # 进行预测，获取模型的输出out，是一个tensor（1*2*480*640）
            out = model(batch_x)

            loss = loss_func(out, batch_y)
            """
            使用item()的原因：如果loss都直接用loss表示，结果是每次迭代，空间占用会增加，直到cpu或者gup爆炸。
            PyTorch 0.4.0版本将Variable和Tensor融合起来，可以视Variable为requires_grad=True的Tensor。
            输出的loss的数据类型是Variable。如果这里直接将loss加起来，系统会认为这里也是计算图的一部分，导致网络会一直延伸变大。
            """
            train_loss += loss.item()

            """
            torch.max(out, d) ，out是模型输出的tensor，d=0时求每列最大值，d=1时求每行最大值
            ”在计算准确率时第一个tensor的values是不需要的，所以只需提取第二个tensor，即这里的[1]“
            还没弄懂这句话什么意思，但取[0]进行训练的话，准确率一直是0
            """
            pred = torch.max(out, 1)[1]

            # 准确率计算，需要除以图片尺寸
            train_correct = (pred == batch_y).sum() / (height * width)
            train_acc += train_correct.item()
            # math.ceil(x) 返回大于等于参数x的最小整数
            print("epoch: %2d/%d batch %3d/%d Train loss: %.3f, Acc: %.3f" %
                  (epoch + 1, epochs, batch, math.ceil(len(train_data) / batch_size),
                   loss.item(), train_correct.item() / len(batch_x)))

            """
            把梯度置零，即把loss关于weight的导数变成0。目前是算一个batch计算一次梯度，然后进行一次梯度更新。
            这里梯度值就是对应偏导数的计算结果,在进行下一次batch梯度计算的时候，前一个batch的梯度计算结果，没有保留的必要了。
            """
            optimizer.zero_grad()
            """
            调用backward，Pytorch的autograd就会自动沿着计算图反向传播计算每一个叶子节点的梯度
            来计算链式法则求导之后计算的结果值
            """
            loss.backward()
            """
            更新参数，即更新权重参数w和b
            """
            optimizer.step()

        scheduler.step()  # 更新学习率
        print("Train loss: %.6f, Acc: %.3f" % (train_loss / (math.ceil(len(train_data) / epochs)),
                                               train_acc / (len(train_data))))

        # 保存模型，每一次epoch的权重模型都保存下来
        if (epoch + 1) % 1 == 0:
            torch.save(model, pth_dir + r"\epoch_" + str(epoch + 1) + ".pth")

if __name__ == "__main__":
    train(r"D:\beam-transfer\pth2")