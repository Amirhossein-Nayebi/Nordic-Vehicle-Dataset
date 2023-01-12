from typing import Any
from typing import List
from typing import Dict
import cv2
import numpy as np


def DrawBoundingBoxWithLabel(
    image: np.ndarray,
    box: List[float],
    color: Any = (128, 128, 128),
    thickness: int = None,
    lineType: int = cv2.LINE_AA,
    label: str = '',
    txtColor: Any = (255, 255, 255)) -> None:

    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    lineWidth = thickness or max(round(sum(image.shape) / 2 * 0.003),
                                 2)  # line width
    cv2.rectangle(image, p1, p2, color, thickness=lineWidth, lineType=lineType)
    if label:
        tf = max(lineWidth - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lineWidth / 3,
                               thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lineWidth / 3,
                    txtColor,
                    thickness=tf,
                    lineType=cv2.LINE_AA)


objClasses = dict()


def LoadCOCOClasses(fileName: str = "./coco_labels.txt") -> None:
    objClasses.clear()
    with open(fileName, 'r') as input_file:
        # Iterate over each line in the input file
        id = 0
        for line in input_file:
            className = line.strip()
            objClasses[id] = className
            id = id + 1


def GetCOCOClassName(classID: int) -> str:
    if len(objClasses) == 0:
        LoadCOCOClasses()
    if classID in objClasses:
        return objClasses[classID]
    else:
        return f"ID {classID} not found!"


def GetBoundingBoxFromPoints(points: List[float]) -> List[float]:
    minX = points[0][0]
    minY = points[0][1]
    maxX = points[0][0]
    maxY = points[0][1]
    for p in points:
        if p[0] < minX:
            minX = p[0]
        if p[0] > maxX:
            maxX = p[0]
        if p[1] < minY:
            minY = p[1]
        if p[1] > maxY:
            maxY = p[1]
    return [minX, minY, maxX, maxY]