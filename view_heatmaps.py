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
    dataset_dir = opt.data_dir
    image_dir = os.path.join(dataset_dir, 'images')
    heatmap_dir = os.path.join(dataset_dir, 'heatmaps_png')
    
    # checks
    if not os.path.isfile(ann_file):
        sys.exit(f"Failed to find annotation file '{ann_file}'!")
    if os.path.splitext(ann_file)[1] != '.xml':
        sys.exit(f"Annotation file is not .xml '{ann_file}'!")
    if not os.path.isdir(image_dir):
        sys.exit(f"Failed to find image directory '{image_dir}'!")
    if not os.path.isdir(heatmap_dir):
        sys.exit(f"Failed to find heatmap .png directory '{heatmap_dir}'!")   

    # getting all names of files from heatmap from selected annotation file
    heatmap_dir_files = os.listdir(heatmap_dir)
    heatmap_files = []
    ann_file_name = os.path.splitext(os.path.basename(ann_file))[0]
    pattern = re.compile(rf'{ann_file_name}-frame\d{{4}}\.png$')
    for heatmap_file in heatmap_dir_files:
        if (os.path.isfile(os.path.join(heatmap_dir, heatmap_file)) and pattern.match(heatmap_file)):
            heatmap_files.append(heatmap_file)

    screen = screeninfo.get_monitors()[0]
    canvas_width = int(screen.width *0.8)
    canvas_height = int(screen.height *0.8)

    border_size = 20

    cell_width = (canvas_width - 3 * border_size) // 2
    cell_height = (canvas_height - 3 * border_size) // 2

    for file in heatmap_files:
        original_image_path = fr"{image_dir}\{file}"
        heatmap_path = fr"{heatmap_dir}\{file}"

        original_img = cv2.resize(cv2.imread(original_image_path), (cell_width, cell_height))
        heatmap_img = cv2.resize(cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE), (cell_width, cell_height))
        colormap_hot_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_HOT)
        colormap_jet_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        overlay_hot_img = cv2.resize(cv2.addWeighted(original_img, 0.5, colormap_hot_img, 0.5, 0), (cell_width, cell_height))
        overlay_jet_img = cv2.resize(cv2.addWeighted(original_img, 0.5, colormap_jet_img, 0.5, 0), (cell_width, cell_height))

        canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

        canvas[border_size:border_size + cell_height, border_size:border_size + cell_width] = original_img  # top-left
        canvas[border_size:border_size + cell_height, 2*border_size + cell_width: 2*border_size + 2*cell_width] = colormap_jet_img# top-right
        canvas[2*border_size + cell_height:2*border_size + 2*cell_height, border_size:border_size + cell_width] = overlay_jet_img # bottom-left
        canvas[2*border_size + cell_height:2*border_size + 2*cell_height, 2*border_size + cell_width: 2*border_size + 2*cell_width] = overlay_hot_img  # bottom-right

        cv2.imshow(f'Image, Heatmap and Overlay', canvas)

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
    parser.add_argument('--data_dir',
                        type=str,
                        help='directory of created dataset',
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