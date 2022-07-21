import torch

"""
yolov5的一个例子
"""
def use_yolov5():
    model = torch.hub.load(r'D:\pycharmprojects\yolov5', 'custom',
                           path=r'D:\pycharmprojects\yolov5\pts\yolov5s.pt', source='local')
    img = r'D:\pycharmprojects\yolov5\data\images\zidane.jpg'
    results = model(img)
    results.show()

if __name__ == "__main__":
    use_yolov5()