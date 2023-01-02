import cv2
import sys
import os


def GetFrameIndex(frameStr: str) -> int:
    try:
        frameIndex = int(frameStr)
    except Exception as e:
        sys.exit(f"Invalid frame!\r\n{e.args[0]}")
    else:
        return frameIndex


if len(sys.argv) < 5:
    sys.exit(
        "Usage:\r\npython extract_frames [video file] [start frame (s)] [end frame (s)] [output folder] [-i]\r\nUse '-i' to specify frame indexes instead of frame times in seconds."
    )

videoFile = sys.argv[1]
if not os.path.isfile(videoFile):
    sys.exit(f"{videoFile} not found!")

startFrame = GetFrameIndex(sys.argv[2])
endFrame = GetFrameIndex(sys.argv[3])
if endFrame < startFrame:
    sys.exit("End frame should be greater than start frame!")

if not os.path.isdir(sys.argv[4]):
    try:
        os.makedirs(sys.argv[4])
    except Exception as e:
        sys.exit(f"Failed to create output directory!\r\n{e.args[0]}")
outDir = sys.argv[4]

useIndex = False
if len(sys.argv) == 6:
    if sys.argv[5].lower().strip() == "-i":
        useIndex = True

vidcap = cv2.VideoCapture(videoFile)

if not useIndex:
    # Get the frame rate of the video
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    # Calculate the frame index from the frame time
    startFrame = int(startFrame * frame_rate)
    endFrame = int(endFrame * frame_rate)

digitsCount = len(str(endFrame))

# Set the position of the video file reader to the desired frame number
vidcap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
success, image = vidcap.read()
frameIndex = startFrame
while success:
    print("Writing frame", frameIndex, "...")
    frameIndexStr = format(frameIndex, f'0{digitsCount}d')
    frameFileName = os.path.join(outDir, f"frame{frameIndexStr}.jpg")
    cv2.imwrite(frameFileName, image)
    success, image = vidcap.read()
    frameIndex += 1
    if frameIndex > endFrame:
        break