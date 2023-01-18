import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from typing import Any
from ultralytics import YOLO
from Util import utility
import cv2
import numpy as np

model = None
m_name = ""

def Detect(img_file:str, model_name: str) -> Any:
    global model, m_name
    if m_name != model_name:
        model_name = model_name.lower() + ".pt"
        model = YOLO(model_name)
        m_name = model_name

    results = model(img_file)
    return results

def DisplayAndSaveResults(imageFileName:str, results, out_file:str):
    image = cv2.imread(imageFileName)
    if image is None:
        sys.exit("Invalid input image!")
    
    utility.DrawYOLOResults(image, results)
    cv2.imwrite(out_file, image)

    cv2.imshow("YOLOv8", image)
    cv2.waitKey(-1)    
    return image

models_names = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to detect cars using YOLOv8.")

    parser.add_argument("-i",
                        "--input_image",
                        help="Input image file",
                        required=True)
    parser.add_argument("-m",
                        "--model_name",
                        help="YOLOv8 model name. Available models:\r\n" +
                        str(models_names),
                        default='yolov8x')
    parser.add_argument("-o",
                        "--output_directory",
                        help="Output directory",
                        default='Results')
    try:
        args = parser.parse_args()
    except Exception as e:
        sys.exit("Failed to parse command line arguments!\r\n" + e.args[0])

    out_file = os.path.join(args.output_directory, "res_" + os.path.basename(args.input_image))
    results = Detect(args.input_image, args.model_name)
    image = DisplayAndSaveResults(args.input_image, results, out_file)