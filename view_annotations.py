import argparse
import os
import sys
import glob
from Util import utility
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2


def main(opt):
    ann_file = opt.annotation_file
    if not os.path.isfile(ann_file):
        sys.exit(f"Failed to find '{ann_file}'!")

    files = glob.glob(os.path.splitext(ann_file)[0] + ".*")
    vid_file = None
    for file in files:
        if os.path.splitext(file)[1].lower() == ".mp4":
            vid_file = file
            break

    if vid_file is None:
        images_dir = os.path.join(os.path.dirname(ann_file), os.path.splitext(os.path.basename(ann_file))[0])
        print(
            f"No video file found! Searching for extracted frames in '{images_dir}' ..."
        )
        if not os.path.isdir(images_dir):
            sys.exit(f"Failed to find '{images_dir}'!")

    # Load the video
    if vid_file is not None:
        vidcap = VideoFileClip(vid_file)
    else:
        # Get list of all files in the directory
        files = os.listdir(images_dir)
        frame_num_len = 0
        frame_ext = ""
        for file in files:
            if '.png' in file.lower():
                frame_num_len = len(os.path.splitext(file)[0].split('_')[1])
                frame_ext = os.path.splitext(file)[1]
                break
        if frame_num_len == 0:
            sys.exit("Can't find frames!")

    boxesPerFrame, width, height = utility.AnnotationBox.GetBoxesFromXMLAnnotationFile(
        ann_file)

    for frameNum, boxes in tqdm(boxesPerFrame.items()):
        if vid_file is not None:
            image = vidcap.get_frame(frameNum / vidcap.fps)
        else:
            file_name = f"frame_{frameNum:0{frame_num_len}d}{frame_ext}"
            frameFilePath = os.path.join(images_dir, file_name)
            image = cv2.imread(frameFilePath)

        for box in boxes:
            box.Draw(image, (0, 255, 0), 2)
        utility.DisplayImage(ann_file, image, -1, 'q')


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        "This Python script is designed to visualize an annotation XML file. \
            \nIt searches for a video file or a directory containing extracted frames with \
            \nthe same name as the annotation file, located alongside the annotation file. \
            \nOnce the script has found the video or frames, it will draw bounding boxes for each frame."
    )
    parser.add_argument('annotation_file',
                        type=str,
                        help='The annotation file.')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
