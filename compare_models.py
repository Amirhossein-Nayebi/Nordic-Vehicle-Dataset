import os
import cv2
import shutil
from tqdm import tqdm
from Util import utility

# test_file = "../datasets/SmartPlane/test.txt"

# test_file_dir = os.path.dirname(test_file)
# test_images_dir = os.path.join(test_file_dir, "test_images")
# os.makedirs(test_images_dir, exist_ok=True)

# with open(test_file, 'r') as img_list:
#     for img_file in tqdm(img_list):
#         img_file = os.path.join(test_file_dir, img_file.strip())
#         dst_file = os.path.join(test_images_dir, os.path.basename(img_file))
#         shutil.copy2(img_file, dst_file)

baseline_folder = "./runs/results/paper/new/detect/yolov5s-test"
comparison_folder = "./runs/results/paper/new/detect/yolov5s_aug-test"

baseline_folder = os.path.join(baseline_folder, 'labels')
comparison_folder = os.path.join(comparison_folder, 'labels')

baseline_files = os.listdir(baseline_folder)
comparison_files = os.listdir(comparison_folder)

for comp_file in comparison_files:
    if comp_file not in baseline_files:
        comp_file = os.path.basename(comp_file)
        print(comp_file)
        img_file = os.path.splitext(comp_file)[0] + ".png"
        img_file = os.path.join("../datasets/SmartPlane/images/", img_file)
        img = cv2.imread(img_file)
        img_height, img_width = img.shape[:2]
        with open(os.path.join(comparison_folder, comp_file), 'r') as file:
            for line in file:
                splits = line.split(' ')
                ann_box = utility.AnnotationBox.CreateFromNormalizedStraightBox(
                    img_width, img_height, float(splits[1]), float(splits[2]),
                    float(splits[3]), float(splits[4]))
                ann_box.Draw(img, color=(0, 255, 255), thickness=3)
        utility.DisplayImage("test", img)
