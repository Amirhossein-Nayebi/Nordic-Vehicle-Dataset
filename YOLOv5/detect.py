import argparse
import sys
from typing import Any
import torch


def Detect(img_file: str, model_name: str) -> Any:
    model = torch.hub.load('ultralytics/yolov5', model_name)
    results = model(img_file)

    return results


models_names = [
    "YOLOv5n",
    "YOLOv5s",
    "YOLOv5m",
    "YOLOv5l",
    "YOLOv5x",
    "YOLOv5n6",
    "YOLOv5s6",
    "YOLOv5m6",
    "YOLOv5l6",
    "YOLOv5x6",
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
                        default='.')
    try:
        args = parser.parse_args()
    except Exception as e:
        sys.exit("Failed to parse command line arguments!\r\n" + e.args[0])

    results = Detect(args.input_image, args.model_name)
    results.save(save_dir=args.output_directory)
    results.print()
    results.show()
