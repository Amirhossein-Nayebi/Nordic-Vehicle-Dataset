import glob
import os
import cv2

path = "./datasets/smart_plane/images/train/*.png"
files = glob.glob(path)

for file in files:
    os.system(
        f'python ./yolov5_source/detect.py --imgsz 1920 --weights .\\runs\\train\\exp3\\weights\\best.pt --source \"{file}\"'
    )
    print(file)