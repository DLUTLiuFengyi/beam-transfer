import os.path
import segmentation.detect as dt

"""
语义分割提取光线
返回掩码图和光线图
"""
def pull_the_beam(in_path, out_path, pth_path=r"D:\pycharmprojects\beam-transfer\pth\simple_net_1.pth"):

    """
    初始方案*语义分割（读照片->语义分割->获得掩码图->语义分割结果图）

    # 原始图
    img = Image.open(in_path).convert("RGB")
    # 执行语义分割，获取分割后的结果图（目标区域为白色，非目标区域为黑色）
    beam_mask = detect.predict(img).convert("L")
    # 新建黑色背景图，用于后面使用透明蒙版混合图像来创建合成图像
    empty = Image.new("RGBA", (img.size), 0)
    # 原图 + 黑底图 + 掩码图 = 语义分割抽取结果图
    beam_img = Image.composite(img, empty, mask=beam_mask)
    beam_img.save(out_path)

    （暂时不用）
    """

    in_path_split = os.path.split(in_path)
    out_path_split = os.path.split(out_path)

    return dt.predict(in_dir=in_path_split[0], in_name=in_path_split[1],
                   out_dir=out_path_split[0], pth_path=pth_path)

"""
光线特效变换
"""
def transform_beam(a_in_path, b_in_path, out_path):

    out_path_split = os.path.split(out_path)
    out_dir = out_path_split[0]
    a_name = os.path.split(a_in_path)[1]
    b_name = os.path.split(b_in_path)[1]

    # 获取a的掩码图和提取出的光线图
    a_mask, a_beam = pull_the_beam(a_in_path, out_path)
    a_mask.save(os.path.join(out_dir, a_name.split(".")[0] + "-mask." + a_name.split(".")[1]))
    a_beam.save(os.path.join(out_dir, a_name.split(".")[0] + "-beam." + a_name.split(".")[1]))
    # 获取b的掩码图和提取出的光线图
    b_mask, b_beam = pull_the_beam(b_in_path, out_path)
    b_mask.save(os.path.join(out_dir, b_name.split(".")[0] + "-mask." + b_name.split(".")[1]))
    b_beam.save(os.path.join(out_dir, b_name.split(".")[0] + "-beam." + b_name.split(".")[1]))
