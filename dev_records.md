
### v0.1

训练日志如下

```txt
D:\anaconda\envs\two\python.exe D:/pycharmprojects/beam-transfer/segmentation/v01/train.py
epoch:  1/10 batch   1/23 Train loss: 0.698577, Acc: 0.296704
epoch:  1/10 batch   2/23 Train loss: 92.348488, Acc: 0.800620
epoch:  1/10 batch   3/23 Train loss: 3.956046, Acc: 0.099114
epoch:  1/10 batch   4/23 Train loss: 0.679531, Acc: 0.854747
epoch:  1/10 batch   5/23 Train loss: 0.676127, Acc: 0.700800
...
Train loss: 6.232891, Acc: 0.780698
Eval Acc (total): 0.8812751913070679
best epoch num: 1
epoch:  2/10 batch   1/23 Train loss: 0.480522, Acc: 0.823761
epoch:  2/10 batch   2/23 Train loss: 0.277048, Acc: 0.925693
epoch:  2/10 batch   3/23 Train loss: 0.519679, Acc: 0.697771
epoch:  2/10 batch   4/23 Train loss: 0.318392, Acc: 0.910694
...
Train loss: 0.458880, Acc: 0.852279
Eval Acc (total): 0.8812751913070679
best epoch num: 1
epoch:  3/10 batch   1/23 Train loss: 0.274486, Acc: 0.902490
epoch:  3/10 batch   2/23 Train loss: 0.316574, Acc: 0.879007
epoch:  3/10 batch   3/23 Train loss: 0.340119, Acc: 0.868186
...
epoch:  9/10 batch   1/23 Train loss: 0.234682, Acc: 0.946718
epoch:  9/10 batch   2/23 Train loss: 0.465669, Acc: 0.830229
epoch:  9/10 batch   3/23 Train loss: 0.324371, Acc: 0.901472
...
Train loss: 0.467991, Acc: 0.880411
Eval Acc (total): 0.8812751913070679
best epoch num: 1
epoch: 10/10 batch   1/23 Train loss: 0.326592, Acc: 0.900635
epoch: 10/10 batch   2/23 Train loss: 0.406694, Acc: 0.859753
epoch: 10/10 batch   3/23 Train loss: 0.356618, Acc: 0.885270
...
Train loss: 0.469020, Acc: 0.880411
Eval Acc (total): 0.8812751913070679
best epoch num: 1

Process finished with exit code 0

```

显然不对
* 验证集的准确率从第一次epoch开始就保持0.8812575不变， 
一直到最后一次迭代
* 即便某次epoch的准确率特别高（88%+），
使用它对应的pth文件预测效果仍可能出现非常不好的情况，甚至没有效果

观察代码，验证集与训练集的准确率计算方式一样，
采用的是minis数字识别那种一图一个标签类别的判断方法，
暂不知是否跟这有关

#### mIoU

准确率指标改成mIoU后，训练时，有一次在epoch2时mIoU飙到90%，其余在45%，然后其他的训练中epoch的mIoU基本都在55%左右

90%那次的epoch的权重模型效果明显比同一次训练、其他epoch，以及其他的训练的模型效果好

#### 模型效果

同一pth文件加载的模型，对图A有效果，对图B基本无任何效果，都是有可能的

### v0.2

对于labelme生成的label.png图片，是单通道，尺寸为H*W，取值只有0和1

网络输入尺寸是 `batch_size * 3 * H * W`，
网络输出尺寸是 `batch_size * class_num * H * W`，
原始label.png成批后的尺寸是 `batch_size * H * W`

PASCAL VOC2012数据集的标签图，也是8位PNG格式，单通道的颜色索引图像，
像素值0代表背景，255代表边界，1~20为20个类别。

调了一通后（尺寸、学习率等），暂时有比较正常的效果

但核心问题：标签图是单通道H*W，网络输出是2通道 1*2*H*W，模型将种类数（2）与单通道标签图内的值（0和1）映射起来的关键点在哪

训练日志（UNet）

```txt
epoch 1/10, epoch_loss: 0.211120, epoch_acc: 0.843043
epoch 1/10, epoch_val_loss: 0.075293, epoch_val_acc: 0.881331
epoch 2/10, epoch_loss: 0.043370, epoch_acc: 0.880315
epoch 2/10, epoch_val_loss: 0.035687, epoch_val_acc: 0.881331
epoch 3/10, epoch_loss: 0.033026, epoch_acc: 0.880315
epoch 3/10, epoch_val_loss: 0.035212, epoch_val_acc: 0.881331
epoch 4/10, epoch_loss: 0.030553, epoch_acc: 0.880315
epoch 4/10, epoch_val_loss: 0.034595, epoch_val_acc: 0.881331
epoch 5/10, epoch_loss: 0.029670, epoch_acc: 0.883347
epoch 5/10, epoch_val_loss: 0.030023, epoch_val_acc: 0.881331
epoch 6/10, epoch_loss: 0.029338, epoch_acc: 0.880315
epoch 6/10, epoch_val_loss: 0.030445, epoch_val_acc: 0.881331
epoch 7/10, epoch_loss: 0.028921, epoch_acc: 0.889089
epoch 7/10, epoch_val_loss: 0.034925, epoch_val_acc: 0.907227
epoch 8/10, epoch_loss: 0.027440, epoch_acc: 0.912107
epoch 8/10, epoch_val_loss: 0.030587, epoch_val_acc: 0.913955
epoch 9/10, epoch_loss: 0.026377, epoch_acc: 0.915043
epoch 9/10, epoch_val_loss: 0.030537, epoch_val_acc: 0.915300
epoch 10/10, epoch_loss: 0.025625, epoch_acc: 0.913229
epoch 10/10, epoch_val_loss: 0.032089, epoch_val_acc: 0.914481

Process finished with exit code 0

```

训练日志（SimpleNet）

```txt
epoch 1/10, epoch_loss: 0.052265, epoch_acc: 0.843822
epoch 1/10, epoch_val_loss: 0.041308, epoch_val_acc: 0.881331
epoch 2/10, epoch_loss: 0.033725, epoch_acc: 0.888004
epoch 2/10, epoch_val_loss: 0.034254, epoch_val_acc: 0.900280
epoch 3/10, epoch_loss: 0.031161, epoch_acc: 0.891744
epoch 3/10, epoch_val_loss: 0.032329, epoch_val_acc: 0.890296
epoch 4/10, epoch_loss: 0.031013, epoch_acc: 0.880770
epoch 4/10, epoch_val_loss: 0.034697, epoch_val_acc: 0.891760
epoch 5/10, epoch_loss: 0.029209, epoch_acc: 0.899353
epoch 5/10, epoch_val_loss: 0.032551, epoch_val_acc: 0.889924
epoch 6/10, epoch_loss: 0.029630, epoch_acc: 0.890089
epoch 6/10, epoch_val_loss: 0.051616, epoch_val_acc: 0.896296
epoch 7/10, epoch_loss: 0.029985, epoch_acc: 0.888342
epoch 7/10, epoch_val_loss: 0.029650, epoch_val_acc: 0.906995
epoch 8/10, epoch_loss: 0.028154, epoch_acc: 0.898412
epoch 8/10, epoch_val_loss: 0.031346, epoch_val_acc: 0.898605
epoch 9/10, epoch_loss: 0.027688, epoch_acc: 0.902436
epoch 9/10, epoch_val_loss: 0.032317, epoch_val_acc: 0.896920
epoch 10/10, epoch_loss: 0.026957, epoch_acc: 0.904248
epoch 10/10, epoch_val_loss: 0.031096, epoch_val_acc: 0.896628

Process finished with exit code 0

```


