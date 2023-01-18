import math
import cv2
import xml.etree.ElementTree as et
import numpy as np
from Util import utility
import YOLOv5.detect as yolo5, YOLOv8.detect as yolo8

videoFile = "./Videos/2022-12-04 Bjenberg 02.MP4"
annotationFile = "./Annotations/CVAT video annotation for 2022-12-04 Bjenberg 02 frame 0 - 2334/annotations.xml"
outFolder = "Result"

yolo5Model = "yolov5x"
yolo8Model = "yolov8x"

vidcap = cv2.VideoCapture(videoFile)
tree = et.parse(annotationFile)
root = tree.getroot()

width = int(root.find("meta/original_size/width").text)
height = int(root.find("meta/original_size/height").text)
allBoxes = root.findall("track/box")

ground_truth_color = (0, 0, 255)
ground_truth_txt_color = (255, 255, 0)

boxesByFrames = {}
for box in allBoxes:
    frameNumber = int(box.attrib["frame"])
    if frameNumber in boxesByFrames:
        boxesByFrames[frameNumber].append(box)
    else:
        boxesByFrames[frameNumber] = [box]

for frameNumber, boxes in boxesByFrames.items():
    # Set the position of the video file reader to the desired frame number
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
    success, image = vidcap.read()

    # Draw ground truth bounding boxes
    for box in boxes:
        if box.attrib["outside"] == "0":
            tlPoint = np.array(
                [float(box.attrib["xtl"]),
                 float(box.attrib["ytl"])])
            brPoint = np.array(
                [float(box.attrib["xbr"]),
                 float(box.attrib["ybr"])])
            trPoint = np.array([brPoint[0], tlPoint[1]])
            blPoint = np.array([tlPoint[0], brPoint[1]])

            c_Point = (tlPoint + brPoint) / 2

            rotation = float(box.attrib["rotation"])
            rot_rad = rotation * math.pi / 180
            cs = math.cos(rot_rad)
            sn = math.sin(rot_rad)
            rot_mat = np.array([[cs, -sn], [sn, cs]])

            p1 = rot_mat @ (tlPoint - c_Point) + c_Point
            p2 = rot_mat @ (trPoint - c_Point) + c_Point
            p3 = rot_mat @ (brPoint - c_Point) + c_Point
            p4 = rot_mat @ (blPoint - c_Point) + c_Point

            points = np.array([p1, p2, p3, p4], np.int32)
            points = points.reshape((-1, 1, 2))
            # print(points)

            cv2.polylines(image, [points], True, ground_truth_color, 1)

            bbox = utility.GetBoundingBoxFromPoints([p1, p2, p3, p4])
            utility.DrawBoundingBoxWithLabel(image,
                                             bbox,
                                             ground_truth_color,
                                             1,
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
