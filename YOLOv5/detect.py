import argparse
import sys
from typing import Any
import torch
import numpy as np
import cv2

model = None
m_name = ""

def Detect(image:np.ndarray, model_name: str) -> Any:
    global model, m_name
    if m_name != model_name:
        model = torch.hub.load('ultralytics/yolov5', model_name.lower())
        m_name = model_name

    results = model(image)
    return results


models_names = [
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
    "yolov5n6",
    "yolov5s6",
    "yolov5m6",
    "yolov5l6",
    "yolov5x6",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to detect cars using YOLOv5.")

    parser.add_argument("-i",
                        "--input_image",
                        help="Input image file",
                        required=True)
    parser.add_argument("-m",
                        "--model_name",
                        help="YOLOv5 model name. Available models:\r\n" +
                        str(models_names),
                        default='yolov5x')
    parser.add_argument("-o",
                        "--output_directory",
                        help="Output directory",
                        default='Results')
    try:
        args = parser.parse_args()
    except Exception as e:
        sys.exit("Failed to parse command line arguments!\r\n" + e.args[0])

    image = cv2.imread(args.input_image)
    results = Detect(image, args.model_name)
    results.save(save_dir=args.output_directory)
    results.print()
    results.show()
