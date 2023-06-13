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

def create_annotations(annotationFile):
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
                    CarAnnotation(box))
            else:
                carsByFrames[frameNumber] = [
                    CarAnnotation(box)
                ]
                
    return carsByFrames, width, height

class CarAnnotation:

    def __init__(self, box: et.Element):
        if "rotation" in box.attrib:
            self.rotation = math.radians(float(box.attrib["rotation"]))
        else:
            self.rotation = math.radians(0)

        tl = np.array([float(box.attrib["xtl"]), float(box.attrib["ytl"])])
        br = np.array([float(box.attrib["xbr"]), float(box.attrib["ybr"])])

        self.center = np.array([((tl[0] + br[0])/2), ((tl[1] + br[1])/2)])
        width = br[0] - tl[0]
        height = br[1] - tl[1]

        self.sigma_x = width/2.8
        self.sigma_y = height/2.8

def main(opt):
    
    # inputs
    ann_dir = opt.ann_dir
    image_dir = opt.image_dir
    heatmap_dir = opt.heatmap_dir

    # checks
    if not os.path.isdir(image_dir):
        sys.exit(f"Failed to find image directory '{image_dir}'!")
    if not os.path.isdir(ann_dir):
        sys.exit(f"Failed to find annotation directory '{ann_dir}'!")
    if not os.path.isdir(heatmap_dir):
        sys.exit(f"Failed to find heatmap directory for output. please create one at '{heatmap_dir}'")   

    # getting all annotation files
    ann_dir_files = os.listdir(ann_dir)
    ann_files = []
    for ann_file in ann_dir_files:
        if os.path.isfile(os.path.join(ann_dir, ann_file)) and os.path.splitext(ann_file)[1] == '.xml':
            ann_files.append(ann_file)
    if (len(ann_files) == 0):
        sys.exit(f"there are no .xml annotation files in annotation directory.")

    #create dictionary of filenames (because there are sometimes different number of prepending zeroes in frame number)
    image_dict = {}
    for filename in os.listdir(image_dir):
        number = filename.split("-")[-1].split(".")[0].split("e")[-1]
        number = str(int(number))
        start = "-".join(filename.split("-")[:-1])
        end = filename.split("-")[-1].split(".")[-1]
        value = f"{start}-{number}"
        image_dict[value] = filename


    # iterate over .xml annotation
    for ann_file in tqdm(ann_files):
        # get boxes per frame
        carsByFrames, width, height = create_annotations(os.path.join(ann_dir, ann_file))

        for frameNum, cars in tqdm(carsByFrames.items()):
            # get correct image
            # old style
            #frame_number =  "{:04d}".format(frameNum)
            #ann_base_file_name = os.path.splitext(ann_file)[0]
            #heatmap_name = fr"{ann_base_file_name}-frame{frame_number}.png"
            # new style
            ann_base_file_name = os.path.splitext(ann_file)[0]
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

                # normalize the heatmap to range between 0 and 255, round it and convert to 8-bit unsigned integer
                heatmap = np.round((heatmap / np.max(heatmap)) * 255).astype(np.uint8)
                # heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # save heatmap to the corresponding path
                if not cv2.imwrite(fr"{heatmap_dir}\{heatmap_name}", heatmap):
                    sys.exit(f"error saving heatmap {heatmap_name}")
            

def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        'This python script creates heatmaps from images and annotation file')
    parser.add_argument('--ann_dir',
                        type=str,
                        help='directory of all the .xml annotation files (files must have same name as images (without -frame0000))',
                        required=True)
    parser.add_argument('--image_dir',
                        type=str,
                        help='directory containing all of the images',
                        required=True)
    parser.add_argument('--heatmap_dir',
                        type=str,
                        help='output directory which will contain all of the heatmaps',
                        required=True)
    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)