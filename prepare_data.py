import argparse
import os
import sys
import glob
from Util import utility
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
from moviepy.video.io.VideoFileClip import VideoFileClip
import imageio
import albumentations as A
import cv2
import data_stats

data_file = "smart_plane.yaml"


def main(opt):

    print()
    video_files = []

    test_videos = opt.test_videos  # Videos which will be used for creating test frames
    train_size = opt.train_size
    val_size = opt.val_size
    test_size = opt.test_size
    sequential = opt.sequential

    if opt.source is not None:  # Source videos are provided, so the frames will be created in the 'image' folder.
        source = os.path.abspath(opt.source)
        is_source_dir = os.path.isdir(source)
        is_source_file = os.path.isfile(source)
        if not is_source_dir and not is_source_file:
            sys.exit(f"There is no directory or file at path '{source}'!")

        if is_source_dir:
            all_files = os.listdir(source)
            video_files = [
                fname for fname in all_files if fname.lower().endswith('.mp4')
            ]
            if len(video_files) == 0:
                sys.exit(
                    f"The input directory '{source}' does not contain any MP4 videos!"
                )
        else:
            video_files = [source]

    # Create data directory if it does not exist.
    # 'images' and 'labels' folders and train, val and test list files will be created in this folder.
    data_dir = os.path.abspath(opt.data_dir)

    if not os.path.isdir(data_dir):
        try:
            os.makedirs(data_dir)
        except Exception as e:
            sys.exit(f"Failed to create data directory '{data_dir}'\r\n{e}")

    # Create images and labels directories if they do not exist.
    imgs_dir = os.path.join(data_dir, "images")
    lbs_dir = os.path.join(data_dir, 'labels')
    try:
        if not os.path.isdir(imgs_dir):
            os.makedirs(imgs_dir)
        if not os.path.isdir(lbs_dir):
            os.makedirs(lbs_dir)
    except Exception as e:
        sys.exit(f"Failed to create output directories!\r\n{e}")

    annotated_videos = []  # List of annotated videos

    if len(video_files) > 0:
        for video_file in tqdm(video_files):
            if is_source_dir:
                video_file_path = os.path.join(source, video_file)
                annotation_file_path = os.path.join(
                    source,
                    os.path.splitext(video_file)[0] + '.xml')
            else:
                video_file_path = video_file
                annotation_file_path = os.path.splitext(
                    video_file_path)[0] + '.xml'

            if not os.path.isfile(annotation_file_path):
                print(f"Annotation file '{annotation_file_path}' not found!")
                continue

            vidcap = VideoFileClip(video_file_path)

            # Get the duration of the video
            duration = vidcap.duration

            # Calculate the total number of frames
            frames_count = int(duration * vidcap.fps)
            frame_num_len = len(str(frames_count))

            # Extract bounding boxes from annotation file
            boxesPerFrame, width, height = utility.AnnotationBox.GetBoxesFromXMLAnnotationFile(
                annotation_file_path)

            videoFileBaseName = os.path.basename(video_file_path)
            annotated_videos.append(videoFileBaseName)

            print(f'Retrieving frames from {videoFileBaseName}...')

            for frameNum, boxes in tqdm(boxesPerFrame.items()):
                frameNumberStr = format(frameNum, f'0{frame_num_len}d')
                frameFilePath = os.path.join(
                    imgs_dir,
                    videoFileBaseName + f"-frame{frameNumberStr}.png")

                if not os.path.isfile(frameFilePath):
                    image = vidcap.get_frame(frameNum / vidcap.fps)
                    cv2.imwrite(frameFilePath, image) # cv2 is much faster than imageio in writing image files
                    # imageio.imwrite(frameFilePath, image) 
                else:
                    print(f'{frameFilePath} exists. Skipping...')

                if len(boxes) > 0:
                    lines = []
                    for box in boxes:
                        yolo_bbox = box.GetYOLOBoundingBox(width, height)
                        if yolo_bbox is None:
                            continue
                        yolo_bbox_str = "0"
                        for elem in yolo_bbox:
                            yolo_bbox_str += f" {elem:.5f}"
                        lines.append(yolo_bbox_str + "\n")

                    labelFilePath = os.path.join(
                        lbs_dir,
                        os.path.splitext(os.path.basename(frameFilePath))[0] +
                        ".txt")

                    with open(labelFilePath, 'w') as file:
                        file.writelines(lines)

    images = glob.glob(os.path.join(os.path.relpath(imgs_dir, '.'), "*.png"))
    # labels = glob.glob(os.path.join(os.path.relpath(lbs_dir, '.'), "*.txt"))

    if len(images) == 0:
        sys.exit(f"No image found in '{data_dir}/images'. \
            \nMake sure each video has a corresponding .xml annotation file.\
            \nIf you have not extracted images from annotated videos yet, please rerun the script with \
                \n--videos_dir option to generate images and labels directories."
                 )

    for i in range(len(images)):
        images[i] = "./images/" + os.path.basename(images[i])

    if test_videos is None:
        if not sequential:
            images_train, images_val_test = train_test_split(
                images, train_size=train_size)

            images_val, images_test = train_test_split(images_val_test,
                                                       test_size=test_size /
                                                       (test_size + val_size))
        else:
            images_train = []
            images_val = []
            images_test = []

            # Normalize train, val, and test sizes to make sure they sum up to one
            sum_sizes = train_size + val_size + test_size
            train_size /= sum_sizes
            val_size /= sum_sizes
            test_size /= sum_sizes

            for annotated_video in annotated_videos:
                all_video_frames = glob.glob(
                    os.path.join(os.path.relpath(imgs_dir, '.'),
                                 f"*{annotated_video}*"))
                for i in range(len(all_video_frames)):
                    all_video_frames[i] = "./images/" + os.path.basename(all_video_frames[i])

                frames_count = len(all_video_frames)
                train_index = 0
                val_index = train_index + int(train_size * frames_count)
                test_index = val_index + int(val_size * frames_count)

                images_train.extend(all_video_frames[train_index:val_index])
                images_val.extend(all_video_frames[val_index:test_index])
                images_test.extend(all_video_frames[test_index:])

    else:
        images_test = []
        images_train_val = []
        for image_path in images:
            is_from_test_video = False
            for test_video in test_videos:
                if os.path.basename(test_video) in image_path:
                    is_from_test_video = True
                    break
            if is_from_test_video:
                images_test.append(image_path)
            else:
                images_train_val.append(image_path)
        images_train, images_val = train_test_split(images_train_val,
                                                    train_size=opt.train_size)

    trainFile = 'train.txt'
    valFile = 'val.txt'
    testFile = 'test.txt'

    print()
    with open(os.path.join(data_dir, trainFile), 'w') as file:
        file.writelines('\n'.join(images_train))
    print(f"'{trainFile}' created in '{data_dir}'.")

    with open(os.path.join(data_dir, valFile), 'w') as file:
        file.writelines('\n'.join(images_val))
    print(f"'{valFile}' created in '{data_dir}'.")

    with open(os.path.join(data_dir, testFile), 'w') as file:
        file.writelines('\n'.join(images_test))
    print(f"'{testFile}' created in '{data_dir}'.")

    data = {
        'path': os.path.abspath(data_dir),  # dataset root dir
        'train': trainFile,
        'val': valFile,
        'test': testFile,

        # Classes
        'names': {
            0: 'car'
        },
    }

    # Write YAML file
    with open(data_file, "w") as file:
        yaml.dump(data, file)

    print(f"'{data_file}' created successfully in '{os.path.abspath('.')}'.\n")

    data_stats.run()

def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        'This python script generates train, validation and test sets for \
            YOLO network using videos and their annotation files.')
    parser.add_argument(
        '--source',
        type=str,
        help=
        'A directory path containing video and corresponding annotation files or a path to a video file. \
            \nIf specified, images and labels data are generated from annotated videos in the data directory. \
            \nIf not, only data lists are updated based on the contents of the data directory.',
        required=False)
    parser.add_argument('--data_dir',
                        type=str,
                        help='Data directory to store generated labeled data',
                        required=True)
    parser.add_argument('--train_size',
                        type=float,
                        help='Train set size ratio',
                        default=0.6)
    parser.add_argument('--val_size',
                        type=float,
                        help='Validation set size ratio',
                        default=0.2)
    parser.add_argument('--test_size',
                        type=float,
                        help='Test set size ratio',
                        default=0.2)
    parser.add_argument(
        '--test_videos',
        type=str,
        nargs="+",
        help=
        'List of test video files separated by spaces. If it is not provided, test data will be created randomly from all videos.',
        required=False)

    parser.add_argument('--sequential',
                        action='store_true',
                        help='Create datasets sequentially not randomly.')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)