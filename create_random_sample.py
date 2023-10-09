import shutil
import os
from tqdm import tqdm
import random
from pathlib import Path
import yaml
import argparse
import sys

def main(opt):
    in_dataset = opt.in_data
    out_dataset = opt.out_data
    in_yaml = opt.smart_plane_path
    data_types = ["train", "val", "test"]
    sample_num = [opt.train, opt.val, opt.test]

    in_images = os.path.join(in_dataset, "images")
    in_heatmaps = os.path.join(in_dataset, "heatmaps")
    in_heatmaps_png = os.path.join(in_dataset, "heatmaps_png")
    in_labels = os.path.join(in_dataset, "labels")

    out_images = os.path.join(out_dataset, "images")
    out_heatmaps = os.path.join(out_dataset, "heatmaps")
    out_heatmaps_png = os.path.join(out_dataset, "heatmaps_png")
    out_labels = os.path.join(out_dataset, "labels")
    out_yaml = os.path.join(out_dataset, "smart_plane.yaml")

    try:
        Path(out_images).mkdir(parents=True, exist_ok=True)
        Path(out_heatmaps).mkdir(parents=True, exist_ok=True)
        Path(out_heatmaps_png).mkdir(parents=True, exist_ok=True)
        Path(out_labels).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        sys.exit(f"Failed to create output directories -- {e}")

    with open(in_yaml) as input_file:
        doc = yaml.safe_load(input_file)
        doc['path'] = out_dataset

        with open(out_yaml, 'w') as output_file:
            yaml.dump(doc, output_file)

    for x, data_type in enumerate(data_types):
        if (sample_num[x] == 0):
            continue

        in_txt = os.path.join(in_dataset, data_type + ".txt")
        out_txt = os.path.join(out_dataset, data_type + ".txt")

        #create sample test file
        with open(in_txt, 'r') as readfile:
            with open(out_txt, 'w') as writefile:
                image_array = readfile.readlines()

                if (len(image_array) < sample_num[x]):
                    sys.exit(f"There are less items in {data_types[x]} dataset ({len(image_array)}) than requested sample size ({sample_num[x]})")
                for image in random.sample(image_array, sample_num[x]):
                    writefile.write(image)

        #copy images and labels
        with open(out_txt, 'r') as testfile:
            for line in tqdm(testfile, desc=f"Copying {data_type} dataset"):
                line = line.rstrip('\n')
                filename = os.path.basename(line)
                base_filename = os.path.splitext(filename)[0]

                image_path = os.path.join(in_images, filename)
                heatmap_path = os.path.join(in_heatmaps, base_filename + ".npy")
                heatmap_png_path = os.path.join(in_heatmaps_png, filename)
                label_path = os.path.join(in_labels, base_filename + ".txt")

                shutil.copy2(image_path, out_images)
                shutil.copy2(heatmap_path, out_heatmaps)
                shutil.copy2(heatmap_png_path, out_heatmaps_png)
                shutil.copy2(label_path, out_labels)

def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        'This python script creates heatmaps from images and annotation file')
    parser.add_argument('--in_data',
                        type=str,
                        help='directory of input dataset',
                        required=True)
    parser.add_argument('--out_data',
                        type=str,
                        help='directory of output sample dataset',
                        required=True)
    parser.add_argument('--smart_plane_path',
                        type=str,
                        help='path to smart_plane.yaml file of input dataset',
                        required=True)
    parser.add_argument('--train',
                        type=int,
                        help='how many images should be in train dataset',
                        required=True)
    parser.add_argument('--val',
                        type=int,
                        help='how many images should be in validation dataset',
                        required=True)
    parser.add_argument('--test',
                        type=int,
                        help='how many images should be in test dataset',
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