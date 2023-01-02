import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
img = '../Extract Frames/2022-12-02_Asjo_01/46-48/frame1180.jpg'
results = model(img)
results.show()
results.save()
results.print()
