import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# print(sys.path)

from ultralytics import YOLO
import cv2
from pathlib import Path
import re
from ExtractFrames import extract_frames
import utility

dir_path = Path(__file__).parent.resolve()

modelNames = [
    # "yolov8n.pt",
    # "yolov8s.pt",
    # "yolov8m.pt",
    # "yolov8l.pt",
    "yolov8x.pt",
]

videoFilePath = Path.joinpath(dir_path, "../Videos/2022-12-02_Asjo_01.MP4")
videoFileName = os.path.basename(videoFilePath)

FrameSlots = [
    # "00:46 - 00:48",
    # "00:53 - 00:57",
    # "02:02 - 02:09",
    # "02:50 - 02:59",
    # "03:32 - 03:37",
    # "03:49 - 03:59",
    # "04:06 - 04:16",
    # "04:25 - 04:31",
    "04:32 - 04:34",
    "06:32 - 06:47",
    "07:17 - 07:29",
    "07:46 - 08:01",
    "08:55 - 08:57",
    "09:22 - 09:30",
    "10:30 - 10:32",
    "10:36 - 10:46",
]

resultFolder = "Results"

for frameSlot in FrameSlots:
    splits = frameSlot.split('-')
    startFrame = splits[0].strip()
    endFrame = splits[1].strip()
    outFolder = "tmp"

    frames = extract_frames.Extract(str(videoFilePath), startFrame, endFrame,
                                    str(outFolder), True)

    for m in modelNames:
        model = YOLO(m)
        for frame in frames:
            # frame = "./bus.jpg"
            image = cv2.imread(frame)
            cv2.imshow("Result", image)
            if cv2.waitKey(1) == ord('q'):
                sys.exit()

            print(frame, m)
            results = model(frame)  # predict on an image
            # print(results)

            if len(results[0]) > 0:
                for res in results[0]:
                    classLabel = int(res[5].item())
                    className = utility.GetCOCOClassName(classLabel)
                    conf = int(round(res[4].item() * 100))
                    box = res[:4].tolist()
                    utility.DrawBoundingBoxWithLabel(
                        image,
                        box, (0, 255, 0),
                        thickness=2,
                        label=f"{className} {conf}%",
                        txtColor=(0, 0, 0))
                frameName = os.path.basename(frame)
                resImgFileName = os.path.join(resultFolder, videoFileName,
                                              frameSlot, m,
                                              frameName).replace(':', '_')
                resImgDir = os.path.dirname(resImgFileName)
                if not os.path.isdir(resImgDir):
                    os.makedirs(resImgDir)
                saveRes = cv2.imwrite(resImgFileName, image)
                cv2.imshow("Result", image)
                if cv2.waitKey(2000) == ord('q'):
                    sys.exit()