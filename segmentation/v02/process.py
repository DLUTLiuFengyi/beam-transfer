import os.path
import segmentation.v01.detect as dt

"""
语义分割提取光线
返回掩码图和光线图
"""
def pull_the_beam(in_path, out_path, pth_path=r"D:\pycharmprojects\beam-transfer\pth\simple_net_1.pth"):

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
