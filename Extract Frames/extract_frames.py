import cv2
import sys
import os


def GetFrameIndex(frameStr: str) -> int:
    splits = frameStr.split(':')
    factor = 1
    frameIndex = 0
    for i in range(len(splits) - 1, -1, -1):
        try:
            frameIndex = frameIndex + factor * int(splits[i])
        except Exception as e:
            sys.exit(f"Invalid frame!\r\n{e.args[0]}")
        factor = factor * 60
    return frameIndex


if len(sys.argv) < 5:
    sys.exit(
        "Usage:\r\npython extract_frames [video file] [start frame] [end frame] [output folder] [-i]\r\nUse '-i' to specify frame indexes instead of frame times in seconds."
    )

videoFile = sys.argv[1]
if not os.path.isfile(videoFile):
    sys.exit(f"{videoFile} not found!")

startFrame = GetFrameIndex(sys.argv[2])
endFrame = GetFrameIndex(sys.argv[3])
if endFrame < startFrame:
    sys.exit("End frame should be greater than start frame!")

outDir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    sys.argv[4], os.path.basename(videoFile),
    sys.argv[2].replace(":", "_") + "-" + sys.argv[3].replace(":", "_"))

if not os.path.isdir(outDir):
    try:
        os.makedirs(outDir)
    except Exception as e:
        sys.exit(
            f"Failed to create output directory '{outDir}'!\r\n{e.args[0]}")

if not os.path.isdir(outDir):
    sys.exit(f"Failed to create output directory '{outDir}'!")

useIndex = False
if len(sys.argv) == 6:
    if sys.argv[5].lower().strip() == "-i":
        useIndex = True
    else:
        print(f"Unknown command line argument: '{sys.argv[5]}'")

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
    frameFileName = os.path.join(outDir, f"frame{frameIndexStr}.png")
    cv2.imwrite(frameFileName, image)
    success, image = vidcap.read()
    frameIndex += 1
    if frameIndex > endFrame:
        break