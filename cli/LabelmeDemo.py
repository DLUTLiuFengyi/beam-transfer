import os
import glob
import shutil
import numpy as np
from PIL import Image

"""
批量执行labelme将json文件转换成标签图的命令
"""
def transfer_json_to_doc(path, ids=[]):
    root_path = path
    if len(ids) > 0:
        for i in ids:
            file = os.path.join(path, str(i) + ".json")
            os.system("labelme_json_to_dataset.exe %s" % (file))
    else:
        json_files = glob.glob(os.path.join(root_path, "*.json"))
        # os.system("conda activate labelme")
        count = 0
        for file in json_files:
            print("count = %d" % count)
            os.system("labelme_json_to_dataset.exe %s" % (file))
            count += 1
        # os.system("conda deactivate")

"""
批量地将执行labelme转换命令后的标签图提取到指定目录下
"""
def pick_up_png_from_doc(root_path, direct_path, ids=[]):
    if len(ids) > 0:
        for i in ids:
            file_path = os.path.join(root_path, str(i) + "_json", "label.png")
            shutil.copy(file_path, os.path.join(direct_path, str(i) + ".png"))
            print(file_path + " successfully moved")
    else:
        for file in os.listdir(root_path):
            file_path = root_path + "\\" + file
            if os.path.isdir(file_path):
                if os.path.exists(file_path + "\\label.png"):
                    shutil.copy(file_path + "\\label.png", direct_path + "\\" + file.split('_')[0] + ".png")
                    print(file + " successfully moved")



if __name__ == "__main__":
    # transfer_json_to_doc(r"D:\beam-transfer\beams")
    # pick_up_png_from_doc(r"D:\beam-transfer\beams", r"D:\beam-transfer\labels")

    # get_pic_shape(r"D:\beam-transfer\labels\68.png") # (766, 1373)

    # transfer_json_to_doc(r"D:\beam-transfer\beams")
    pick_up_png_from_doc(r"D:\beam-transfer\beams", r"D:\beam-transfer\labels")