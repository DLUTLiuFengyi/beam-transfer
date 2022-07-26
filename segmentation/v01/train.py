import numpy
import torch
import math
from torchvision import transforms
from dataset import BeamDataset
import segmentation.v01.net as snet
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from segmentation.v01.utils import judge_cuda
from segmentation.v01.metrics import get_hist, getIoU, getMIoU

"""
模型训练
"""
def train(pth_dir=r"D:\beam-transfer\pth"):
    # os.makedirs(pth_dir, exist_ok=True)
    height = 480
    width = 640
    batch_size=8
    epochs = 10
    learning_rate = 0.04
    # 加载训练集
    # height与width太大的话，GPU显存不够用
    train_data = BeamDataset(height=height, width=width, name_file="train.txt", transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    # 加载验证集
    eval_data = BeamDataset(height=height, width=width, name_file="val.txt", transform=transforms.ToTensor())
    eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size)

    model = snet.SimpleNet()
    if judge_cuda():
        model.cuda()

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.CrossEntropyLoss()


    # 开始训练
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            # 获取训练所用的真实图x和标签图y
            if judge_cuda():
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            # 进行预测，获取模型的输出out，是一个tensor（batch_size*2*480*640）
            out = model(batch_x)

            loss = loss_func(out, batch_y)
            # 损失值累加（用item()原因是新版本pytorch把Variable和Tensor类合并，输出的loss是Variable类型）
            train_loss += loss.item()

            # 把梯度置零，即把loss关于weight的导数变成0。目前是算一个batch计算一次梯度，然后进行一次梯度更新
            optimizer.zero_grad()
            # 沿着计算图反向传播计算每一个叶子节点的梯度，来计算链式法则求导之后计算的结果值
            loss.backward()
            # 更新参数，即更新权重参数w和b
            optimizer.step()

        scheduler.step()  # 更新学习率
        print("Train stage epoch {}/{}, loss_epoch: {}".format(epoch + 1, epochs, train_loss / batch))

        # 模型验证
        model.eval()
        eval_loss = 0
        IoU_epoch = numpy.zeros(2)
        mIoU_epoch = 0
        for batch, (batch_x, batch_y) in enumerate(eval_loader):
            if judge_cuda():
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()

            _, pred = torch.max(out, 1)

            hist = get_hist(pred.cpu().detach().numpy(), batch_y.cpu().detach().numpy(), 2)
            IoU = getIoU(hist)
            IoU_epoch += IoU

            mIoU = getMIoU(IoU)
            mIoU_epoch += mIoU

            # print("epoch {}/{}, val batch {}/{}".format(epoch+1, epochs, batch, math.ceil(len(eval_data) / batch_size)))
            # print("IoU: {}, mIoU: {}".format(IoU, mIoU))

        print("Eval stage epoch {}/{}, loss_epoch: {}".format(epoch + 1, epochs, eval_loss / batch))
        print("Eval stage IoU_epoch: {}\nmIoU_epoch: {}".format(IoU_epoch / batch, mIoU_epoch / batch)) # mIoU可以类比准确率，越接近100%越好

        # 保存模型，每一次epoch的权重模型都保存下来
        if (epoch + 1) % 1 == 0:
            torch.save(model, pth_dir + r"\epoch_" + str(epoch + 1) + ".pth")

if __name__ == "__main__":
    train(pth_dir=r"D:\beam-transfer\pth2")