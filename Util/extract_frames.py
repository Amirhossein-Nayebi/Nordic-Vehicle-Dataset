import cv2
import sys
import os
import argparse


def GetFrameIndex(frameStr: str) -> int:
    splits = frameStr.split(':')
    factor = 1
    frameInSeconds = 0
    for i in range(len(splits) - 1, -1, -1):
        try:
            frameInSeconds = frameInSeconds + factor * int(splits[i])
        except Exception as e:
            sys.exit(f"Invalid frame!\r\n{e.args[0]}")
        factor = factor * 60
    return frameInSeconds


def Extract(videoFile: str, startFrame: str, endFrame: str, outDir: str,
            inSeconds: bool) -> list:

    extractedFrames = []

    videoFile = videoFile.strip()
    startFrame = startFrame.strip()
    endFrame = endFrame.strip()
    outDir = outDir.strip()

    if not os.path.isfile(videoFile):
        sys.exit(f"'{videoFile}' not found!")

    startFrameInSeconds = GetFrameIndex(startFrame)
    endFrameInSeconds = GetFrameIndex(endFrame)
    if endFrameInSeconds < startFrameInSeconds:
        sys.exit("End frame should be greater than start frame!")

    outDir = os.path.join(
        outDir, os.path.basename(videoFile),
        startFrame.replace(":", "_") + "-" + endFrame.replace(":", "_"))

    if not os.path.isdir(outDir):
        try:
            os.makedirs(outDir)
        except Exception as e:
            sys.exit(
                f"Failed to create output directory '{outDir}'!\r\n{e.args[0]}"
            )

    vidcap = cv2.VideoCapture(videoFile)

    if inSeconds:
        # Get the frame rate of the video
        frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
        # Calculate the frame index from the frame time
        startFrameNumber = int(startFrameInSeconds * frame_rate)
        endFrameNumber = int(endFrameInSeconds * frame_rate)
    else:
        startFrameNumber = startFrameInSeconds
        endFrameNumber = endFrameInSeconds

    digitsCount = len(str(endFrameNumber))

    # Set the position of the video file reader to the desired frame number
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, startFrameNumber)
    success, image = vidcap.read()
    frameNumber = startFrameNumber
    while success:
        print("Writing frame", frameNumber, "...")
        frameNumberStr = format(frameNumber, f'0{digitsCount}d')
        frameFileName = os.path.join(outDir, f"frame{frameNumberStr}.png")
        cv2.imwrite(frameFileName, image)
        extractedFrames.append(frameFileName)
        success, image = vidcap.read()
        frameNumber += 1
        if frameNumber > endFrameNumber:
            break
        
    return extractedFrames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to extract frames from a video file.")

    parser.add_argument("-f", "--file", help="Video file", required=True)
    parser.add_argument(
        "-s",
        "--start",
        help="Start frame. If '-i' is not specified it is in seconds, \
            e.g. 01:23 else it is the start frame number",
        required=True)
    parser.add_argument(
        "-e",
        "--end",
        help="End frame. If '-i' is not specified it is in seconds, \
            e.g. 01:23 else it is the end frame number",
        required=True)
    parser.add_argument("-o", "--output", help="Output folder", required=True)
    parser.add_argument(
        "-i",
        "--index",
        help="Start and end frames are frame numbers not seconds.",
        type=bool,
        required=False)
    try:
        args = parser.parse_args()
    except Exception as e:
        sys.exit("Failed to parse command line arguments!\r\n" + e.args[0])

    Extract(args.file, args.start, args.end, args.output, args.index is None)