''' 
Given a video file and CVAT annotation xml file, this script generates
data for training YOLO networks.
'''
import math
import xml.etree.ElementTree as et
import numpy as np
import cv2

videoFile = "./Videos/2022-12-04 Bjenberg 02.MP4"
annotationFile = "./Annotations/CVAT video annotation for 2022-12-04 Bjenberg 02 frame 0 - 2334/annotations.xml"
outFolder = "Result"

vidcap = cv2.VideoCapture(videoFile)

tree = et.parse(annotationFile)
root = tree.getroot()

width = int(root.find("meta/original_size/width").text)
height = int(root.find("meta/original_size/height").text)

boxes = root.findall("track/box")
for box in boxes:
    if box.attrib["outside"] == "0":
        frameNumber = int(box.attrib["frame"])
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

        # Set the position of the video file reader to the desired frame number
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
        success, image = vidcap.read()


        points = np.array([p1, p2, p3, p4], np.int32)
        points = points.reshape((-1, 1, 2))
        # print(points)

        cv2.polylines(image, [points], True, (0, 255, 0), 1)
        # cv2.drawContours(image,[[[tuple(p1.astype(int)), tuple(p2.astype(int)), tuple(p3.astype(int)), tuple(p4.astype(int))]]], 0, (0, 255, 0), 2)

        cv2.imshow("test", image)
        if cv2.waitKey(-1) == ord('q'):
            break
