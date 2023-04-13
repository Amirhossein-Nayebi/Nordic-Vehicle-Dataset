import os
import cv2
import shutil
from tqdm import tqdm
from Util import utility

# Open the file in read mode
with open('../datasets/SmartPlane/test.txt', 'r') as file:
    # Read all lines into a list
    test_label_files = file.readlines()

yolov5s_detected_test_frames = os.listdir(
    "./runs/results/paper/new/detect/yolov5s-test/labels")
yolov5s_aug_detected_test_frames = os.listdir(
    "./runs/results/paper/new/detect/yolov5s_aug-test/labels")
yolov8s_detected_test_frames = os.listdir(
    "./runs/results/paper/new/detect/yolov8s-test/labels")
yolov8s_aug_detected_test_frames = os.listdir(
    "./runs/results/paper/new/detect/yolov8s_aug-test/labels")

for test_label_file in test_label_files:
    test_label_file = test_label_file[len('./images/'):-5] + ".txt"
    if test_label_file in yolov5s_detected_test_frames:
        continue
    if test_label_file in yolov5s_aug_detected_test_frames:
        continue
    if test_label_file in yolov8s_detected_test_frames:
        continue
    if test_label_file in yolov8s_aug_detected_test_frames:
        continue
    img_file = os.path.splitext(test_label_file)[0] + ".png"
    img_file = os.path.join("../datasets/SmartPlane/images/", img_file)
    img = cv2.imread(img_file)
    utility.DisplayImage("test", img)