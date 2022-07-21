import torch
import backgroundremover.u2net.u2net as u2n
from PIL import Image
from torchvision import transforms

"""
参考backgroundremover的代码，现在用不到
"""

"""
读取模型网络和权重参数
"""
def get_model():
    net = u2n.U2NET(3, 1)
    pt_path = r"D:\beam-transfer\u2net.pth"
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(pt_path))
        net.to(torch.device("cuda"))
    else:
        net.load_state_dict(torch.load(pt_path, map_location="cpu"))

    net.eval()
    return net

"""
归一化
"""
def normalize(d):
    max = torch.max(d)
    min = torch.min(d)
    return (d - min) / (max - min)

"""
执行预测
"""
def predict(img):

    model = get_model()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img)

    with torch.no_grad(): # 只做预测的话，无需让torch反向传播
        if torch.cuda.is_available():
            img_input = torch.cuda.FloatTensor(img_tensor.unsqueeze(0).cuda().float())
        else:
            img_input = torch.FloatTensor(img_tensor.unsqueeze(0).float())

        d1, d2, d3, d4, d5, d6, d7 = model(img_input)

        predict = normalize(d1[:, 0, :, :]).squeeze().cpu().detach().numpy()
        predict_img = Image.fromarray(predict * 255).convert("RGB")

        return predict_img


