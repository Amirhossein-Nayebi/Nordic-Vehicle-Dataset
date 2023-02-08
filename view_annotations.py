import argparse
import os
import sys
import glob
from Util import utility
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip


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
        sys.exit(f"Failed to find a video file for '{ann_file}'! \
                \nMake sure that there is a video file with the same name along with the annotation file."
                 )

    # Load the video
    vidcap = VideoFileClip(vid_file)

    boxesPerFrame, width, height = utility.AnnotationBox.GetBoxesFromXMLAnnotationFile(
        ann_file)

    for frameNum, boxes in tqdm(boxesPerFrame.items()):
        image = vidcap.get_frame(frameNum / vidcap.fps)
        for box in boxes:
            box.Draw(image, (0, 255, 0), 2)
        utility.DisplayImage(ann_file, image, -1, 'q')


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description='\nThis python script visualize an annotation xml file.\
            \nIt looks for a video file with the same name as the annotation file and draw the bounding boxes frame by frame.'
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
