import cv2
import numpy as np
import os
import argparse
import sys
import math
from tqdm import tqdm
from Util import utility
from scipy.stats import multivariate_normal
import xml.etree.ElementTree as et
import glob
import re 

def create_annotations(annotationFile, sigma_border_range):
    tree = et.parse(annotationFile)
    root = tree.getroot()

    width = int(root.find("meta//original_size/width").text)
    height = int(root.find("meta//original_size/height").text)
    allBoxes = root.findall("track/box")

    carsByFrames = {}
    for box in allBoxes:
        if box.attrib["outside"] == "0":
            frameNumber = int(box.attrib["frame"])
            if frameNumber in carsByFrames:
                carsByFrames[frameNumber].append(
                    CarAnnotation(box, sigma_border_range))
            else:
                carsByFrames[frameNumber] = [
                    CarAnnotation(box, sigma_border_range)
                ]
                
    return carsByFrames, width, height

class CarAnnotation:

    def __init__(self, box: et.Element, sigma_border_range):
        if "rotation" in box.attrib:
            self.rotation = math.radians(float(box.attrib["rotation"]))
        else:
            self.rotation = math.radians(0)

        tl = np.array([float(box.attrib["xtl"]), float(box.attrib["ytl"])])
        br = np.array([float(box.attrib["xbr"]), float(box.attrib["ybr"])])

        self.center = np.array([((tl[0] + br[0])/2), ((tl[1] + br[1])/2)])
        width = br[0] - tl[0]
        height = br[1] - tl[1]

        self.sigma_x = width / sigma_border_range
        self.sigma_y = height / sigma_border_range

def main(opt):
    
    # inputs
    source_dir = opt.source
    dataset_dir = opt.data_dir
    sigma_border_range = opt.sigma_border_range

    new_image_dir = os.path.join(dataset_dir, 'images')
    new_heatmap_dir = os.path.join(dataset_dir, 'heatmaps')
    new_heatmap_image_dir = os.path.join(dataset_dir, 'heatmaps_png')

    # checks
    if not os.path.isdir(source_dir):
        sys.exit(f"Failed to find source directory with annotations '{source_dir}'!")
    if not os.path.isdir(dataset_dir):
        sys.exit(f"Failed to find dataset directory '{dataset_dir}'!")
    if not os.path.isdir(new_image_dir):
        sys.exit(f"Failed to find image directory '{new_image_dir}' inside dataset directory '{dataset_dir}'!")

    # getting all annotation files
    source_dir_files = os.listdir(source_dir)
    source_files = []
    for source_file in source_dir_files:
        if os.path.isfile(os.path.join(source_dir, source_file)) and os.path.splitext(source_file)[1] == '.xml':
            source_files.append(source_file)
    if (len(source_files) == 0):
        sys.exit(f"there are no .xml annotation files in annotation directory.")

    # create heatmap directories
    if not os.path.isdir(new_heatmap_dir):
        try:
            os.mkdir(new_heatmap_dir)
        except Exception as e:
            sys.exit(f"Failed to create heatmap .npy directory '{new_heatmap_dir} -- {e}")
    if not os.path.isdir(new_heatmap_image_dir):
        try:
            os.mkdir(new_heatmap_image_dir)
        except Exception as e:
            sys.exit(f"Failed to create heatmap .png directory '{new_heatmap_image_dir} -- {e}")

    # create dictionary of filenames (because there are sometimes different number of prepending zeroes in frame number)
    image_dict = {}
    for filename in os.listdir(new_image_dir):
        number = filename.split("-")[-1].split(".")[0].split("e")[-1]
        number = str(int(number))
        start = "-".join(filename.split("-")[:-1])
        end = filename.split("-")[-1].split(".")[-1]
        value = f"{start}-{number}"
        image_dict[value] = filename

    # iterate over .xml annotation
    for source_file in tqdm(source_files):
        # get boxes per frame
        carsByFrames, width, height = create_annotations(os.path.join(source_dir, source_file), sigma_border_range)

        for frameNum, cars in tqdm(carsByFrames.items()):
            # get correct image
            ann_base_file_name = os.path.splitext(source_file)[0]
            key = f"{ann_base_file_name}-{frameNum}"
            heatmap_name = image_dict[key]
            
            # grid of coords
            xv, yv = np.meshgrid(np.arange(width), np.arange(height))
            # create blank heatmap with same size as images
            heatmap = np.zeros((height, width))

            if (len(cars) != 0):
                car_heatmaps = []

                for car in cars:
                    # rotation matrix
                    rotation_matrix = np.array([[np.cos(car.rotation), -np.sin(car.rotation)], [np.sin(car.rotation), np.cos(car.rotation)]])
                    # initial covariance matrix (before rotation)
                    cov = np.array([[car.sigma_x**2, 0], [0, car.sigma_y**2]])
                    # rotate the covariance matrix
                    cov_rotated = rotation_matrix @ cov @ rotation_matrix.T

                    # create a 2d gaussian centered at (x,y)
                    gaussian_kernel = multivariate_normal([car.center[0], car.center[1]], cov_rotated)
                    car_heatmap = gaussian_kernel.pdf(np.dstack((xv, yv)))
                    # normalize individual heatmap and add it to the overall heatmap
                    car_heatmaps.append((car_heatmap / np.max(car_heatmap)))

                # # variation 1: add all normalized heatmaps together
                for car_heatmap in car_heatmaps:
                    heatmap += car_heatmap
                # variation 2: get only the highest value of a pixel from all heatmaps 
                # heatmap = np.maximum.reduce(car_heatmaps)

                # final normalization
                heatmap /= np.max(heatmap)

                # save .npy heatmap as float 32
                heatmap = heatmap.astype(np.float32)
                heatmap_raw_name = os.path.splitext(heatmap_name)[0] + ".npy"
                np.save(fr"{new_heatmap_dir}\{heatmap_raw_name}",heatmap)

                # normalize the heatmap to range between 0 and 255 convert to 8-bit unsigned integer
                # save heatmap to the corresponding path
                heatmap_img = (heatmap* 255).astype(np.uint8)
                if not cv2.imwrite(fr"{new_heatmap_image_dir}\{heatmap_name}", heatmap_img):
                    sys.exit(f"error saving heatmap {heatmap_name}")
            

def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        'This python script creates heatmaps from images and annotation file')
    parser.add_argument('--source',
                        type=str,
                        help='directory of all the .xml annotation files (files must have same name as images (without -frame0000))',
                        required=True)
    parser.add_argument('--data_dir',
                        type=str,
                        help='directory of created dataset by prepare_data.py',
                        required=True)
    parser.add_argument('--sigma_border_range',
                        type=float,
                        help='defines the sigma value at the range of border of a car',
                        default=3.25)
    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)