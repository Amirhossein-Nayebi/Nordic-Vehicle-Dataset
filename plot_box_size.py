import argparse
import os
import sys
import numpy as np
import csv

from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from Util import utility
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
from matplotlib import pyplot as plt


def main(opt):
    source = os.path.abspath(opt.source)
    dir_source = os.path.isdir(source)
    file_source = os.path.isfile(source)
    if not dir_source and not file_source:
        sys.exit(f"There is no directory or file at path '{source}'!")

    if dir_source:
        all_files = os.listdir(source)
        ann_files = [
            os.path.join(source, fname) for fname in all_files if fname.lower().endswith('.xml')
        ]
        if len(ann_files) == 0:
            sys.exit(
                f"The input directory '{source}' does not contain any xml annotation file!"
            )
    else:
        ann_files = [source]

    video_data_file = 'video_data.csv'
    if not os.path.isfile(video_data_file):
        sys.exit(f"'{video_data_file}' not found!")

    for ann_file in ann_files:
        max_height = -1
        with open(video_data_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if row[0].lower() + ".xml" == os.path.basename(ann_file).lower():
                    height_str = row[1].replace('m', '').strip()
                    height_splits = height_str.split('-')
                    max_height = float(height_splits[-1])
                    break

        if max_height < 0:
            sys.exit(f"Failed to get height data from '{video_data_file}' for '{ann_file}'")

        boxesPerFrame, width, height = utility.AnnotationBox.GetBoxesFromXMLAnnotationFile(
            ann_file)

        frames = []
        sizes = []
        for frameNum, boxes in tqdm(boxesPerFrame.items()):
            if len(boxes) == 0:
                continue
            # print(frameNum)
            size = 0
            for box in boxes:
                size += box.GetDiagonalLength()
            size /= len(boxes)
            frames.append(frameNum)
            sizes.append(1 / size)

        frames = np.array(frames)
        sizes = np.array(sizes)

        # Fit a robust polynomial regression model using RANSAC
        degree = 4
        ransac = RANSACRegressor(make_pipeline(PolynomialFeatures(degree),
                                            LinearRegression()),
                                residual_threshold=2.0,
                                random_state=0,
                                min_samples=20)
        ransac.fit(frames[:, np.newaxis], sizes)

        min_frame = frames.min()
        max_frame = frames.max()

        # Predict using the RANSAC model
        line_x = np.linspace(min_frame, max_frame, 100)
        line_y = ransac.predict(line_x[:, np.newaxis])
        max_y = line_y.max()

        height_scale = max_height / max_y
        heights = line_y * height_scale

        # Plot the results
        fig, ax1 = plt.subplots()
        # ax1.plot(line_x, line_y, color='red', linewidth=3)
        ax1.plot(line_x, heights, linewidth=3, color='red', label='Estimated Height (m)')
        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("Flight Height (m)")
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.scatter(frames,
                    sizes,
                    marker='.',
                    facecolors='none',
                    edgecolors='blue',
                    s=10, label='Reciprocal of Box Diagonal Length (1 / Pixels)')
        x_min, x_max = ax2.get_xlim()
        ax2.plot([x_min, x_max], [max_y, max_y], linestyle='--', label=f"Max Height = {max_height} m")
        
        ax2.set_ylabel("Reciprocal of Box Diagonal Length (1 / pixels)")
        y1, y2 = ax2.get_ylim()
        y1 *= height_scale
        y2 *= height_scale
        ax1.set_ylim(y1, y2)

        ax2.legend(loc='upper right')
        title = os.path.basename(ann_file)
        plt.title(title)
        fig.set_size_inches(8, 5)

        out_dir = opt.out_dir
        if not os.path.isdir(out_dir):
             os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, title + ".png"))
    plt.show()


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        'This Python script visualizes the annotated box sizes in each frame. \
            \nIt gets an annotation xml file and plots the mean box size in each frame vs frame number. \
            \nIt is useful to study variation of flight heigh.')
    parser.add_argument('source',
                        type=str,
                        help='An annotation file or a directory containing annotation files. \
                            \nIn case the source argument is a directory, one plot will be drawn per each xml annotation file.')
    parser.add_argument('--out_dir',
                        type=str,
                        help='Directory where the plots will be saved. (Default = plots)',
                        default='plots')
                            
    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
