import math
import cv2
import numpy as np
from Util import utility
import YOLOv5.detect as yolo5, YOLOv8.detect as yolo8

videoFile = "./Videos/2022-12-04 Bjenberg 02.MP4"
annotationFile = "./Annotations/CVAT video annotation for 2022-12-04 Bjenberg 02 frame 0 - 2334/annotations.xml"
outFolder = "Result"

yolo5Model = "yolov5x"
yolo8Model = "yolov8x"

ground_truth_color = (0, 0, 255)
ground_truth_txt_color = (255, 255, 0)

vidcap = cv2.VideoCapture(videoFile)

boxesByFrames, width, height = utility.AnnotationBox.GetBoxesFromXMLAnnotationFile(
    annotationFile)

for frameNumber, boxes in boxesByFrames.items():
    # Set the position of the video file reader to the desired frame number
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
    success, image = vidcap.read()

    # Draw ground truth bounding boxes
    for box in boxes:
        box.Draw(image,
                 color=ground_truth_color,
                 thickness=1,
                 label="Ground Truth",
                 txtColor=ground_truth_txt_color)

    yolo5Results = yolo5.Detect(image, yolo5Model)
    utility.DrawYOLOResults(image, yolo5Results.pred, yolo5Model + " ")

    cv2.imwrite("tmp.png", image)
    yolo8Results = yolo8.Detect("tmp.png", yolo8Model)
    utility.DrawYOLOResults(image, yolo8Results, yolo8Model + " ")

    cv2.imshow("test", image)
    if cv2.waitKey(-1) == ord('q'):
        break
