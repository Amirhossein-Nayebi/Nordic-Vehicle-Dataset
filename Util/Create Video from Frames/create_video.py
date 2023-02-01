import cv2
import os

codec = cv2.VideoWriter_fourcc(*'XVID')
fps = 25
frame_size = (1920, 1080)
output_file = 'output.avi'

out = cv2.VideoWriter(output_file, codec, fps, frame_size)

firstFrameFile = "./ExtractFrames/Results/2022-12-02_Asjo_01.MP4/10_30-10_32/frame15750.png"
lastFrameFile = "./ExtractFrames/Results/2022-12-02_Asjo_01.MP4/10_30-10_32/frame15800.png"

dir = os.path.dirname(firstFrameFile)

firstFrame = int(
    os.path.basename(firstFrameFile).replace("frame", "").replace(".png", ""))
lastFrame = int(
    os.path.basename(lastFrameFile).replace("frame", "").replace(".png", ""))

for fNum in range(firstFrame, lastFrame + 1):
    fileName = os.path.join(dir, f"frame{str(fNum)}.png")
    frame = cv2.imread(fileName)
    out.write(frame)
    print(int((fNum - firstFrame) * 100 / (lastFrame - firstFrame)), '%')

out.release()