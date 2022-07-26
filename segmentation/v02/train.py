import torch
import math
from torchvision import transforms
from dataset import BeamDataset
import segmentation.v02.net as snet
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from segmentation.v02.beam_utils import judge_cuda
from torch.optim import lr_scheduler


"""
模型训练

pth_dir: pth文件保存路径

这个训练的代码有问题
batch_x表示的是一批图片，其中每张图片的类别不是一维的，因为这是语义分割，不是手写数字识别
"""
def train(pth_dir=r"D:\beam-transfer\pth"):
    # os.makedirs(pth_dir, exist_ok=True)
    height = 480
    width = 640
    batch_size=8
    epochs = 10
    learning_rate = 0.001
    # 加载训练集
    # height与width太大的话，GPU显存不够用
    train_data = BeamDataset(height=height, width=width, name_file="train.txt")
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    # 加载验证集
    val_data = BeamDataset(height=height, width=width, name_file="val.txt")
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)

    model = snet.SimpleNet()
    # model = snet.UNet()
    if judge_cuda():
        model.cuda()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # 损失函数
    loss_func = nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0

        model.train()

        for batch, (batch_x, batch_y) in enumerate(train_loader):
            # 获取训练所用的真实图x和标签图y
            if judge_cuda():
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            # 进行预测
            out = model(batch_x)

            # 计算损失值
            loss = loss_func(out, batch_y)

            # 把梯度置零，即把loss关于weight的导数变成0。目前是算一个batch计算一次梯度，然后进行一次梯度更新
            optimizer.zero_grad()
            # 沿着计算图反向传播计算每一个叶子节点的梯度，来计算链式法则求导之后计算的结果值
            loss.backward()
            # 更新梯度参数，即更新权重参数w和b
            optimizer.step()

            # 计算这次epoch的损失值和准确率
            with torch.no_grad(): # 前向传播计算，不记录梯度
                out = torch.argmax(out, dim=1)
                correct += (out == batch_y).sum().item()
                total += batch_y.size(0)
                running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / (total * height * width)

        print("epoch %d/%d, epoch_loss: %f, epoch_acc: %f" % (epoch + 1, epochs, epoch_loss, epoch_acc))

        # 模型验证
        val_correct = 0
        val_total = 0
        val_running_loss = 0
        model.eval()
        with torch.no_grad():
            for batch, (batch_x, batch_y) in enumerate(val_loader):
                if judge_cuda():
                    batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
                else:
                    batch_x, batch_y = Variable(batch_x), Variable(batch_y)
                out = model(batch_x)
                loss = loss_func(out, batch_y)
                out = torch.argmax(out, dim=1)
                val_correct += (out == batch_y).sum().item()
                val_total += batch_y.size(0)
                val_running_loss += loss.item()

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / (val_total * height * width)

        print("epoch %d/%d, epoch_val_loss: %f, epoch_val_acc: %f" % (epoch + 1, epochs, epoch_val_loss, epoch_val_acc))

        # 保存模型，每一次epoch的权重模型都保存下来
        if (epoch + 1) % 1 == 0:
            torch.save(model, pth_dir + r"\epoch_" + str(epoch + 1) + ".pth")

if __name__ == "__main__":
    train(pth_dir=r"D:\beam-transfer\pth3")