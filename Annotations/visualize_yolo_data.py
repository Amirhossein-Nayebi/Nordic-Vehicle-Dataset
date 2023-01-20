import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import Util.utility as util
import glob
import cv2
from tqdm import tqdm

inFolder = "Result"

search_pattern = '*.png'

file_list = glob.glob(os.path.join(inFolder, search_pattern))

for img_file_path in tqdm(file_list):
    image = cv2.imread(img_file_path)
    img_height, img_width = image.shape[:2]
    label_file_path = os.path.splitext(img_file_path)[0] + ".txt"
    if os.path.isfile(label_file_path):
        with open(label_file_path, 'r') as file:
            for line in file:
                splits = line.split()
                cx = float(splits[1])
                cy = float(splits[2])
                w = float(splits[3])
                h = float(splits[4])
                box = util.AnnotationBox.CreateFromNormalizedStraightBox(img_width, img_height, cx, cy, w, h)
                box.Draw(image, (0, 255, 0), 2)
            cv2.imshow("Result", image)
            if cv2.waitKey(-1) == ord('q'):
                exit()
