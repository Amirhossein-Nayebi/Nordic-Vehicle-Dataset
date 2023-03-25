import os
import cv2
from Util import utility

baseline_folder = "./runs/results/paper/new/detect/yolov5s"
comparison_folder = "./runs/results/paper/new/detect/yolov8s"

baseline_folder = os.path.join(baseline_folder, 'labels')
comparison_folder = os.path.join(comparison_folder, 'labels')

baseline_files = os.listdir(baseline_folder)
comparison_files = os.listdir(comparison_folder)

for comp_file in comparison_files:
    if comp_file not in baseline_files:
        print(comp_file)
        img_file = os.path.basename(comp_file)
        img_file = os.path.splitext(img_file)[0]
        _index = img_file.rfind('_')
        frame_str = img_file[_index:]
        frame_str = frame_str.replace('_', '-frame') + ".png"
        img_file = img_file[:_index] + frame_str
        img_file = os.path.join("../datasets/SmartPlane/images/", img_file)
        img = cv2.imread(img_file)
        img_height, img_width = img.shape[:2]
        with open(os.path.join(comparison_folder, comp_file), 'r') as file:
            for line in file:
                splits = line.split(' ')
                ann_box = utility.AnnotationBox.CreateFromNormalizedStraightBox(
                    img_width, img_height, float(splits[1]), float(splits[2]),
                    float(splits[3]), float(splits[4]))
                ann_box.Draw(img)
        utility.DisplayImage("test", img)