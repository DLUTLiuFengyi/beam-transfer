import torch
import math
from torchvision import transforms
from dataset import BeamDataset
import segmentation.v01.net as snet
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from segmentation.v01.beam_utils import judge_cuda


"""
模型训练

pth_dir: pth文件保存路径

这个训练的代码有问题
batch_x表示的是一批图片，其中每张图片的类别不是一维的，因为这是语义分割，不是手写数字识别
"""
def train(pth_dir=r"D:\beam-transfer\pth"):
    # os.makedirs(pth_dir, exist_ok=True)
    height = 600
    width = 800
    batch_size=8
    epochs = 10
    learning_rate = 0.04
    # 加载训练集
    # height与width太大的话，GPU显存不够用
    train_data = BeamDataset(height=height, width=width, name_file="train.txt", transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    # 加载验证集
    eval_data = BeamDataset(height=height, width=width, name_file="eval.txt", transform=transforms.ToTensor())
    eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size)

    model = snet.SimpleNet()
    if judge_cuda():
        model.cuda()

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    # BeamDataset的__getitem__方法中target有的取值会为-1，这里不ignore的话会报错
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)

    # 记录准确率最高的是哪一次epoch的模型
    best_pth_count = 0
    best_pth_acct = 0

    # 开始训练
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

            # 进行预测，获取模型的输出out，是一个tensor（batch_size*2*480*640）
            out = model(batch_x)

            loss = loss_func(out, batch_y)
            # 损失值累加（用item()原因是新版本pytorch把Variable和Tensor类合并，输出的loss是Variable类型）
            train_loss += loss.item()

            # 分别找出tensor每行中的最大值，并返回索引（即为对应的预测种类的数字）
            # torch.max返回两个值，第一个（_）是具体的预测概率值，第二个（pred）是预测的种类
            _, pred = torch.max(out, 1)

            # 准确率计算，需要除以图片尺寸
            train_correct = (pred == batch_y).sum() / (height * width)
            train_acc += train_correct.item()
            # math.ceil(x) 返回大于等于参数x的最小整数
            print("epoch: %2d/%d batch %3d/%d Train loss: %.6f, Acc: %.6f" %
                  (epoch + 1, epochs, batch+1, math.ceil(len(train_data) / batch_size),
                   loss.item(), train_correct.item() / len(batch_x)))

            # 把梯度置零，即把loss关于weight的导数变成0。目前是算一个batch计算一次梯度，然后进行一次梯度更新
            optimizer.zero_grad()
            # 沿着计算图反向传播计算每一个叶子节点的梯度，来计算链式法则求导之后计算的结果值
            loss.backward()
            # 更新参数，即更新权重参数w和b
            optimizer.step()

        scheduler.step()  # 更新学习率
        print("Train loss: %.6f, Acc: %.6f" % (train_loss / (math.ceil(len(train_data) / epochs)),
                                               train_acc / (len(train_data))))

        # 模型验证
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in eval_loader:
            if judge_cuda():
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            _, pred = torch.max(out, 1)
            eval_correct = (pred == batch_y).sum() / (height * width)
            eval_acc += eval_correct.item()
        print("Eval Acc (total): " + str(eval_acc / len(eval_data)))

        if best_pth_acct < eval_acc:
            best_pth_acct = eval_acc
            best_pth_count = epoch + 1

        # 保存模型，每一次epoch的权重模型都保存下来
        if (epoch + 1) % 1 == 0:
            torch.save(model, pth_dir + r"\epoch_" + str(epoch + 1) + ".pth")

        print("best epoch num: %s" % str(best_pth_count))

if __name__ == "__main__":
    train(pth_dir=r"D:\beam-transfer\pth2")