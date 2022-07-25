import os
import glob
import shutil

"""
批量执行labelme将json文件转换成标签图的命令

ids: 要转换的png的文件名，为空则表示处理全部
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
批量地将执行labelme转换命令后的标签图复制到指定目录下

ids: 要复制的png的文件名
"""
def copy_png_to_label_doc(root_path, direct_path, ids=[]):
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
                    # 复制粘贴到指定目录，注意如果目录之前已有该标签图，则需要在复制前手动删除一下
                    shutil.copy(file_path + "\\label.png", direct_path + "\\" + file.split('_')[0] + ".png")
                    print(file + " successfully moved")



if __name__ == "__main__":
    # transfer_json_to_doc(r"D:\beam-transfer\train\beams")
    # copy_png_to_label_doc(r"D:\beam-transfer\train\beams", r"D:\beam-transfer\train\labels")

    # transfer_json_to_doc(r"D:\beam-transfer\train\beams", [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,
    #                                                  2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,
    #                                                  2021,2022,2023,2024,2025])
    copy_png_to_label_doc(r"D:\beam-transfer\train\beams", r"D:\beam-transfer\train\labels",
                          [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,
                           2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,
                           2021,2022,2023,2024,2025])