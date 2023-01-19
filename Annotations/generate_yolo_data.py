''' 
Given a video file and CVAT annotation xml file, this script generates
data for training YOLO networks.
'''

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import Util.utility as util
import cv2

videoFile = "./Videos/2022-12-04 Bjenberg 02.MP4"
annotationFile = "./Annotations/CVAT video annotation for 2022-12-04 Bjenberg 02 frame 0 - 2334/annotations.xml"
outFolder = "Result"

if not os.path.isdir(outFolder):
    os.makedirs(outFolder)

if not os.path.isdir(outFolder):
    sys.exit("Failed to create output folder!")

videoFileBaseName = os.path.basename(videoFile)

boxesPerFrame, width, height = util.AnnotationBox.GetBoxesFromXMLAnnotationFile(
    annotationFile)

vidcap = cv2.VideoCapture(videoFile)

counter = 0
n = len(boxesPerFrame)

for frameNum, boxes in boxesPerFrame.items():
    # Set the position of the video file reader to the desired frame number
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
    success, image = vidcap.read()

    frameNumberStr = format(frameNum, f'0{6}d')
    frameFilePath = os.path.join(
        outFolder, videoFileBaseName + f"-frame{frameNumberStr}.png")

    cv2.imwrite(frameFilePath, image)

    if len(boxes) > 0:
        lines = []
        for box in boxes:
            yolo_bbox = box.GetYOLOBoundingBox(width, height)
            yolo_bbox_str = "0"
            for elem in yolo_bbox:
                yolo_bbox_str += f" {elem:.3f}"
            lines.append(yolo_bbox_str + "\n")

        labelFilePath = os.path.splitext(frameFilePath)[0] + ".txt"
        with open(labelFilePath, 'w') as file:
            file.writelines(lines)
    counter += 1
    print(int(counter * 100 / n), "%")
