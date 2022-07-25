import os.path
import torch
import cv2
from torch.autograd import Variable
from PIL import Image
from beam_utils import normalize_array, get_tensor_from_img_by_cv2, judge_cuda

"""
预测，输入原始图片，输出模型输出（tensor类型）
"""
def predict(in_dir=r"D:\beam-transfer\try", in_name=r"1008.png",
            out_dir=r"D:\beam-transfer\out", pth_path=r"D:\beam-transfer\pth\epoch_10.pth"):
    model = torch.load(pth_path)
    model.eval()
    model.cuda()

    img_path = os.path.join(in_dir, in_name)
    # 测试图片转成tensor（目前训练数据集是用cv2读取的，所以这里用cv2的方式效果更好）
    img_tensor = get_tensor_from_img_by_cv2(img_path)
    if judge_cuda():
        out = model(Variable(img_tensor.cuda()))
    else:
        out = model(Variable(img_tensor))
    print("predict out size: " + str(out.size()))

    out_array = change_tensor_to_array_1(out)
    print("out_array.size(): " + str(out_array.shape))

    # array_to_img_1(in_dir, in_name, out_dir, out_array[0])
    array_to_img_2(in_dir, in_name, out_dir, out_array[0])
    # return array_to_img_3(in_dir, in_name, out_dir, out_array[0])

"""
模型输出的tensor转成numpy的array，方式1
"""
def change_tensor_to_array_1(out):
    out_array = out.cpu().detach().numpy().squeeze(0)
    print(out_array[0])
    return out_array

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
    out_img = Image.fromarray(normalize_array(array)).convert("L")
    # "100.png" -> "100-out.png"
    out_name = in_name.split(".")[0] + "-out." + in_name.split(".")[1]
    out_img.save(os.path.join(out_dir, out_name))

    # 原始图
    img_origin = Image.open(os.path.join(in_dir, in_name))
    # img_origin = img_origin.resize((640, 480), Image.ANTIALIAS)
    # 黑底图，用于蒙版合成
    empty = Image.new("RGBA", (img_origin.size), 0)
    # 蒙版合成
    result = Image.composite(img_origin, empty, out_img)
    # "100.png" -> "100-result.png"
    result_name = in_name.split(".")[0] + "-result." + in_name.split(".")[1]
    result.save(os.path.join(out_dir, result_name))

"""
array传成图片，方式3
返回分割结果（掩码图）和蒙版合成图（光线）
"""
def array_to_img_3(in_dir, in_name, out_dir, array):
    # 掩码图
    mask_img = Image.fromarray(normalize_array(array)).convert("L")
    # 原始图
    img_origin = Image.open(os.path.join(in_dir, in_name))
    # 黑底图，用于蒙版合成
    empty_img = Image.new("RGBA", (img_origin.size), 0)
    # 蒙版合成
    beam_img = Image.composite(img_origin, empty_img, mask_img)
    return mask_img, beam_img

if __name__ == "__main__":
    predict(in_dir=r"D:\beam-transfer\try", in_name=r"1006.png",
            out_dir=r"D:\beam-transfer\out", pth_path=r"D:\beam-transfer\pth2\epoch_10.pth")