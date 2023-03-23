Move the `augmentations.py` and `augment.py` files to the `site-packages/yolov5/utils` and `site-packages/ultralytics/yolo/data` directories respectively.

To enable Albumentation augmentations change the `p` parameter in the `Albumentations' class from 0.0 to 0.7 in both files. 