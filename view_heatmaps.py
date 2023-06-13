import os
import re
import numpy as np
import cv2
import sys
import screeninfo
import argparse 

def main(opt):
    # inputs
    ann_file = opt.ann_file
    image_dir = opt.image_dir
    heatmap_dir = opt.heatmap_dir

    # checks
    if not os.path.isfile(ann_file):
        sys.exit(f"Failed to find annotation file '{ann_file}'!")
    if os.path.splitext(ann_file)[1] != '.xml':
        sys.exit(f"Annotation file is not .xml '{ann_file}'!")
    if not os.path.isdir(image_dir):
        sys.exit(f"Failed to find image directory '{image_dir}'!")
    if not os.path.isdir(heatmap_dir):
        sys.exit(f"Failed to find heatmap directory '{heatmap_dir}'!")   

    # getting all names of files from heatmap from selected annotation file
    heatmap_dir_files = os.listdir(heatmap_dir)
    heatmap_files = []
    ann_file_name = os.path.splitext(os.path.basename(ann_file))[0]
    pattern = re.compile(rf'{ann_file_name}-frame\d{{4}}\.png$')
    for heatmap_file in heatmap_dir_files:
        if (os.path.isfile(os.path.join(heatmap_dir, heatmap_file)) and pattern.match(heatmap_file)):
            heatmap_files.append(heatmap_file)

    screen = screeninfo.get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height
    image_width = screen_width // 2
    image_height = screen_height // 2

    for file in heatmap_files:

        original_image_path = fr"{image_dir}\{file}"
        heatmap_path = fr"{heatmap_dir}\{file}"

        original_img = cv2.resize(cv2.imread(original_image_path), (image_width, image_height))
        heatmap_img = cv2.resize(cv2.imread(heatmap_path), (image_width, image_height))
        overlay_img = cv2.resize(cv2.addWeighted(original_img, 0.5, heatmap_img, 0.5, 0), (image_width, image_height))

        canvas = 255 * np.ones((screen_height, screen_width, 3), dtype=np.uint8)

        canvas[:image_height, :image_width] = original_img  # Top-left
        canvas[:image_height, image_width:] = heatmap_img  # Top-right
        canvas[image_height:, image_width//2:image_width//2*3] = overlay_img  # Bottom

        cv2.imshow('Image Set', canvas)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()
    cv2.destroyAllWindows()


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        'This python script shows heatmaps. Press anything to view next image or \'q\' to exit')
    parser.add_argument('--ann_file',
                        type=str,
                        help='path to .xml annotation file you want to view',
                        required=True)
    parser.add_argument('--image_dir',
                        type=str,
                        help='directory containing all of the images',
                        required=True)
    parser.add_argument('--heatmap_dir',
                        type=str,
                        help='directory containing all of the heatmaps',
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