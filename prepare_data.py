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

data_file = "smart_plane.yaml"


def main(opt):

    print()
    video_files = []

    if opt.source is not None:
        source = os.path.abspath(opt.source)
        dir_source = os.path.isdir(source)
        file_source = os.path.isfile(source)
        if not dir_source and not file_source:
            sys.exit(f"There is no directory or file at path '{source}'!")

        if dir_source:
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

    data_dir = os.path.abspath(opt.data_dir)
    if not os.path.isdir(data_dir):
        try:
            os.makedirs(data_dir)
        except Exception as e:
            sys.exit(f"Failed to create data directory '{data_dir}'\r\n{e}")
    imgs_dir = os.path.join(data_dir, "images")
    lbs_dir = os.path.join(data_dir, 'labels')
    try:
        if not os.path.isdir(imgs_dir):
            os.makedirs(imgs_dir)
        if not os.path.isdir(lbs_dir):
            os.makedirs(lbs_dir)
    except Exception as e:
        sys.exit(f"Failed to create output directories!\r\n{e}")

    if len(video_files) > 0:
        for video_file in tqdm(video_files):
            if dir_source:
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

            boxesPerFrame, width, height = utility.AnnotationBox.GetBoxesFromXMLAnnotationFile(
                annotation_file_path)

            videoFileBaseName = os.path.basename(video_file_path)
            print(f'Retrieving frames from {videoFileBaseName}...')

            for frameNum, boxes in tqdm(boxesPerFrame.items()):
                frameNumberStr = format(frameNum, f'0{frame_num_len}d')
                frameFilePath = os.path.join(
                    imgs_dir,
                    videoFileBaseName + f"-frame{frameNumberStr}.png")
                
                if not os.path.isfile(frameFilePath):
                    image = vidcap.get_frame(frameNum / vidcap.fps)
                    imageio.imwrite(frameFilePath, image)
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
        # labels[i] = "./labels/" + os.path.basename(labels[i])

    images_train, images_val_test, labels_train, labels_val_test = train_test_split(
        images, images, train_size=opt.train_size)

    images_val, images_test, labels_val, labels_test = train_test_split(
        images_val_test,
        labels_val_test,
        test_size=opt.test_size / (opt.test_size + opt.val_size))

    trainFile = 'train.txt'
    valFile = 'val.txt'
    testFile = 'test.txt'

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
    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    # Usage: import generate_data; generate_data.run(videos_dir=..., data_dir=..., train_size=0.6, val_size=0.2, test_size=0.2)
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)