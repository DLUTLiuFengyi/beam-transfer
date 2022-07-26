import numpy as np
import cv2

"""
获取每行的混淆矩阵
"""
def get_fast_hist(row_pred, row_label, class_num=2):
    mask = (row_label >= 0) & (row_label < class_num)
    # print("row_label:{} row_image:{}".format(row_label[mask], row_pred[mask]))
    hist = np.bincount(class_num * row_label[mask].astype(int) + row_pred[mask], minlength=class_num ** 2).reshape(class_num, class_num)
    return hist

"""
获取混淆矩阵
"""
def get_hist(batch_pred, batch_label, class_num=2):
    hist = np.zeros((class_num, class_num))
    for pred, label in zip(batch_pred, batch_label):
        for row_pred, row_label in zip(pred, label):
            hist += get_fast_hist(row_pred.flatten(), row_label.flatten(), class_num)
    return hist

"""
获取IoU指标
"""
def getIoU(hist):
    # print("hist.shape: {}".format(hist.shape)) # (2, 2)
    IoU = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    return IoU

"""
获取mIoU指标
"""
def getMIoU(IoU):
    return np.mean(IoU)
