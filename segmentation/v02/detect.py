import os.path

import numpy
import torch
import cv2
from torch.autograd import Variable
from PIL import Image
from segmentation.v02.beam_utils import judge_cuda
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

"""
预测
"""
def predict(in_dir=r"D:\beam-transfer\try", in_name=r"1008.png",
            out_dir=r"D:\beam-transfer\out", pth_path=r"D:\beam-transfer\pth\epoch_10.pth"):
    model = torch.load(pth_path)
    model.eval()
    model.cuda()

    height = 480
    width = 640

    img_path = os.path.join(in_dir, in_name)
    img = Image.open(img_path).convert("RGB")
    trans = transforms.Compose([
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor_trans = trans(img)
    img_tensor = img_tensor_trans.unsqueeze(0)
    if judge_cuda():
        out = model(Variable(img_tensor.cuda()))
    else:
        out = model(Variable(img_tensor))
    print("predict out shape: " + str(out.shape)) # 1*2*H*W，2是要识别的种类数（背景和光线）
    print(out)
    # argmax函数，对每列取极大值，变成 H*W
    out_array = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    print(out_array)
    print("out_array shape: " + str(out_array.shape)) # H*W
    print("numpy.unique(out_array): " + str(numpy.unique(out_array))) # [0, 1]，代表现在的2D图像一共有0（背景）和1（光线）这两种类别的像素

    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor_trans.permute(1, 2, 0).numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(out_array)
    plt.show()


if __name__ == "__main__":
    predict(in_dir=r"D:\beam-transfer\try", in_name=r"100.png",
            out_dir=r"D:\beam-transfer\out", pth_path=r"D:\beam-transfer\pth2\epoch_10.pth")