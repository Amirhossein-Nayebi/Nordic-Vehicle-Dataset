import os

baseline_folder = "./runs/results/paper/new/detect/yolov5s"
comparison_folder = "./runs/results/paper/new/detect/yolov8s"

baseline_files = os.listdir(os.path.join(baseline_folder, 'labels'))
comparison_files = os.listdir(os.path.join(comparison_folder, 'labels'))

for comp_file in comparison_files:
    if comp_file not in baseline_files:
        print(comp_file)
        img_file = os.path.basename(comp_file)
        img_file = os.path.splitext(img_file)[0] + ".png"
        img_file = os.path.join("../datasets/SmartPlane/images/", img_file)