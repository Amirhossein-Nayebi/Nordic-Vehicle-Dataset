from typing import Any, Tuple
from typing import List, Dict
import cv2
import numpy as np
import sys
import os
import xml.etree.ElementTree as et
import math
import yaml


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
                '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF',
                '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF',
                'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def WriteDataYAMLFile(data_dir, train_file, val_file, test_file, data_file):
    data = {
        'path': os.path.abspath(data_dir),  # dataset root dir
        'train': train_file,
        'val': val_file,
        'test': test_file,

        # Classes
        'names': {
            0: 'car'
        },
    }

    # Write YAML file
    with open(data_file, "w") as file:
        yaml.dump(data, file)

    print(f"'{data_file}' created successfully in '{os.path.abspath('.')}'.\n")


def GetFramesInfo(frames_dir: str):
    # Get list of all files in the directory
    files = os.listdir(frames_dir)
    frame_num_len = 0
    frame_ext = ""
    for file in files:
        ext = os.path.splitext(file)[1]
        if '.png' == ext.lower():
            frame_num_len = len(os.path.splitext(file)[0].split('_')[1])
            frame_ext = ext
            break
    return frame_num_len, frame_ext


# Displays image and wait for the specified delay (ms).
# Terminates the execution if 'exitChar' is pressed during the delay.
def DisplayImage(title: str,
                 image: np.ndarray,
                 delay: int = 0,
                 exitChar: str = 'q') -> None:
    cv2.imshow(title, image)
    if cv2.waitKey(delay) == ord(exitChar):
        sys.exit()


# Saves image with specified file path.
# It creates path structure if it does not exist.
def SaveImage(filePath: str, image: np.ndarray) -> bool:
    dirPath = os.path.dirname(filePath)
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)
    saveRes = cv2.imwrite(filePath, image)
    return saveRes


def DrawYOLOResults(image: np.ndarray, results, label_prefix=''):
    if len(results[0]) > 0:
        for *box, conf, cls in results[0]:  # xyxy, confidence, class
            className = GetCOCOClassName(int(cls))
            color = colors(cls)
            textColor = (255 - color[0], 255 - color[1], 255 - color[2])
            DrawBoundingBoxWithLabel(
                image,
                box,
                color,
                thickness=2,
                label=f"{label_prefix}{className} {int(conf * 100)}%",
                txtColor=textColor)


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


class AnnotationBox:

    def __init__(self, xtl: float, ytl: float, xbr: float, ybr: float,
                 rotation: float) -> None:
        '''Initializes an AnnotationBox with top-left and bottom-right points'''
        self.__tlPoint = np.array([xtl, ytl])
        self.__brPoint = np.array([xbr, ybr])
        self.__rotation = rotation
        self.UpdateInternalData()

    def CreateFromXMLElement(box: et.Element):
        if "rotation" in box.attrib:
            rotation = float(box.attrib["rotation"])
        else:
            rotation = 0

        return AnnotationBox(float(box.attrib["xtl"]),
                             float(box.attrib["ytl"]),
                             float(box.attrib["xbr"]),
                             float(box.attrib["ybr"]), rotation)

    def CreateFromNormalizedStraightBox(img_width, img_height, centerX,
                                        centerY, box_width, box_height):
        '''Creates an AnnotationBox with normalized box center and size data (YOLO format)'''
        xtl = (centerX - box_width / 2) * img_width
        ytl = (centerY - box_height / 2) * img_height
        xbr = (centerX + box_width / 2) * img_width
        ybr = (centerY + box_height / 2) * img_height
        return AnnotationBox(xtl, ytl, xbr, ybr, 0)

    def UpdateInternalData(self) -> None:
        self.__trPoint = np.array([self.__brPoint[0], self.__tlPoint[1]])
        self.__blPoint = np.array([self.__tlPoint[0], self.__brPoint[1]])
        self.__c_Point = (self.__tlPoint + self.__brPoint) / 2

        rot_rad = self.__rotation * math.pi / 180
        cs = math.cos(rot_rad)
        sn = math.sin(rot_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])

        self.__p1 = rot_mat @ (self.__tlPoint -
                               self.__c_Point) + self.__c_Point
        self.__p2 = rot_mat @ (self.__trPoint -
                               self.__c_Point) + self.__c_Point
        self.__p3 = rot_mat @ (self.__brPoint -
                               self.__c_Point) + self.__c_Point
        self.__p4 = rot_mat @ (self.__blPoint -
                               self.__c_Point) + self.__c_Point

        self.__points = np.array([self.__p1, self.__p2, self.__p3, self.__p4],
                                 np.int32)
        self.__points = self.__points.reshape((-1, 1, 2))

    def GetStraightBoundingBox(self) -> List[float]:
        points = [self.__p1, self.__p2, self.__p3, self.__p4]
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

    def GetDiagonalLength(self):
        distance = np.sqrt(np.sum((self.__tlPoint - self.__brPoint)**2))
        return distance

    def GetYOLOBoundingBox(self, width, height) -> List[float]:
        straight_bbox = self.GetStraightBoundingBox()

        if straight_bbox[0] < 0: straight_bbox[0] = 0
        if straight_bbox[1] < 0: straight_bbox[1] = 0
        if straight_bbox[2] < 0: straight_bbox[2] = 0
        if straight_bbox[3] < 0: straight_bbox[3] = 0

        if straight_bbox[0] >= width: straight_bbox[0] = width - 1
        if straight_bbox[2] >= width: straight_bbox[2] = width - 1
        if straight_bbox[1] >= height: straight_bbox[1] = height - 1
        if straight_bbox[3] >= height: straight_bbox[3] = height - 1

        yolo_bbox = [(straight_bbox[0] + straight_bbox[2]) / 2 / width,
                     (straight_bbox[1] + straight_bbox[3]) / 2 / height,
                     (straight_bbox[2] - straight_bbox[0]) / width,
                     (straight_bbox[3] - straight_bbox[1]) / height]

        if yolo_bbox[0] < 0 or yolo_bbox[0] > 1 or yolo_bbox[
                1] < 0 or yolo_bbox[1] > 1 or yolo_bbox[2] < 0 or yolo_bbox[
                    2] > 1 or yolo_bbox[3] < 0 or yolo_bbox[3] > 1:
            print("Warning - Invalid bounding box!")
        return yolo_bbox

    def GetBoxesFromXMLAnnotationFile(
        annotationFile: str
    ) -> Tuple[Dict[int, List["AnnotationBox"]], int, int]:
        tree = et.parse(annotationFile)
        root = tree.getroot()

        width = int(root.find("meta//original_size/width").text)
        height = int(root.find("meta//original_size/height").text)
        allBoxes = root.findall("track/box")

        boxesByFrames = {}
        for box in allBoxes:
            if box.attrib["outside"] == "0":
                frameNumber = int(box.attrib["frame"])
                if frameNumber in boxesByFrames:
                    boxesByFrames[frameNumber].append(
                        __class__.CreateFromXMLElement(box))
                else:
                    boxesByFrames[frameNumber] = [
                        __class__.CreateFromXMLElement(box)
                    ]
        return boxesByFrames, width, height

    def Draw(self,
             image: np.ndarray,
             color: Any = (128, 128, 128),
             thickness: int = 1,
             lineType: int = cv2.LINE_AA,
             label: str = '',
             txtColor: Any = (255, 255, 255)) -> None:

        cv2.polylines(image, [self.__points], True, color, thickness)

        straight_bbox = self.GetStraightBoundingBox()

        DrawBoundingBoxWithLabel(
            image,
            straight_bbox,
            color,
            1,
            lineType,
            label=label,
            txtColor=txtColor,
        )
