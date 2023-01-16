import argparse
import sys
import cv2
import YOLOv5, YOLOv8

def Detect(imgFilePath: str, detector: str, outDir: str):
    detector = detector.lower()
    if detector == 'yolov5':
        pass
    elif detector == 'yolov8':
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to detect cars in bird view images.")

    parser.add_argument("-i", "--image", help="Input image", required=True)
    parser.add_argument(
        "-d",
        "--detector",
        help="Car detector. Accepted detectors: 'yolov5', 'yolov8'", 
        required=True)
    parser.add_argument("-o",
                        "--out_dir",
                        help="Output directory",
                        default="./")

    try:
        args = parser.parse_args()
    except Exception as e:
        sys.exit("Failed to parse command line arguments!\r\n" + e.args[0])

    Detect(args.image, args.detector, args.out_dir)