import os.path
import torch
import cv2
from torch.autograd import Variable
from PIL import Image
from segmentation.v01.utils import normalize_array, get_tensor_from_img_by_cv2, judge_cuda
from torchvision import transforms

HEIGHT = 480
WIDTH = 640

"""
预测，输入原始图片，输出模型输出（tensor类型）
"""
def predict(in_dir=r"D:\beam-transfer\try", in_name=r"1008.png",
            out_dir=r"D:\beam-transfer\out", pth_path=r"D:\beam-transfer\pth\epoch_10.pth"):

    # 加载模型
    model = torch.load(pth_path)
    model.eval()
    if judge_cuda():
        model.cuda()

    # 图片 -> tensor
    img_path = os.path.join(in_dir, in_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img_transfer = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = img_transfer(img).unsqueeze(0)  # unsqueeze(0)作用：3*480*640 -> 1*3*480*640

    # 模型输入的tensor -> 模型输出的tensor
    if judge_cuda():
        out = model(Variable(img_tensor.cuda()))
    else:
        out = model(Variable(img_tensor))
    print("predict out size: " + str(out.size())) # torch.Size([1, 2, 790, 1057])

    # tensor -> numpy
    out_array = out.cpu().detach().numpy().squeeze(0)
    print(out_array)
    print("out_array.size(): " + str(out_array.shape)) # (2, 790, 1057)

    # numpy -> 图片
    # array_to_img_1(in_dir, in_name, out_dir, out_array[0])
    array_to_img_2(in_dir, in_name, out_dir, out_array[0])
    # return array_to_img_3(in_dir, in_name, out_dir, out_array[0])

"""
array转成图片，方式1
"""
def array_to_img_1(in_dir, in_name, out_dir, array):
    img = cv2.imread(os.path.join(in_dir, in_name))
    out_img = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    cv2.imshow("in", img)
    cv2.imshow("out", out_img)
    cv2.waitKey(0)

"""
array转成图片，方式2
"""
def array_to_img_2(in_dir, in_name, out_dir, array):
    # 模型输出图
    out_img = Image.fromarray(normalize_array(array)).convert("L")
    # 文件名 "100.png" -> "100-out.png"
    out_name = in_name.split(".")[0] + "-out." + in_name.split(".")[1]
    out_img.save(os.path.join(out_dir, out_name))

    # 原始图
    img_origin = Image.open(os.path.join(in_dir, in_name))
    img_origin = img_origin.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

    # 黑底图，用于蒙版合成
    empty_img = Image.new("RGBA", (img_origin.size), 0)

    # 蒙版合成
    result = Image.composite(img_origin, empty_img, out_img)

    # 文件名 "100.png" -> "100-result.png"
    result_name = in_name.split(".")[0] + "-result." + in_name.split(".")[1]
    result.save(os.path.join(out_dir, result_name))

"""
array传成图片，方式3
返回分割结果（掩码图）和蒙版合成图（光线）
"""
def array_to_img_3(in_dir, in_name, out_dir, array):
    # 模型输出图
    out_img = Image.fromarray(normalize_array(array)).convert("L")

    # 原始图
    img_origin = Image.open(os.path.join(in_dir, in_name))
    img_origin = img_origin.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

    # 黑底图，用于蒙版合成
    empty_img = Image.new("RGBA", (img_origin.size), 0)

    # 蒙版合成
    result = Image.composite(img_origin, empty_img, out_img)

    return out_img, result

if __name__ == "__main__":
    predict(in_dir=r"D:\beam-transfer\try", in_name=r"1007.png",
            out_dir=r"D:\beam-transfer\out", pth_path=r"D:\pycharmprojects\beam-transfer\pth\v01_simple_net_5.pth")