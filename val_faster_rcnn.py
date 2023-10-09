import os
import random
import torch, detectron2
import yaml
from tqdm import tqdm
import cv2
import numpy as np
import json
 
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
 
#INPUTS
data_info_file = "/home/home/thesis/dataset/processed_rpi_sample_100/smart_plane.yaml"
model_weight_path = "/home/home/thesis/model/detectron2/faster_rcnn/model_final.pth"

#LOGGER SETUP
setup_logger()
  
#DATASET LOAD
def load_dataset(data_info_file: str, dataset_type: str):
    # data_info_file is the yolo yaml file defining the data
    # dataset_type should be 'train', 'val' or 'test'
    with open(data_info_file) as file:
        data_info = yaml.safe_load(file)
 
    root_path = data_info['path']
    img_list_file = os.path.join(root_path, data_info[dataset_type])
 
    dataset_dicts = []
 
    with open(img_list_file, 'r') as img_list:
        for idx, img_path in tqdm(enumerate(img_list)):
            record = {}
 
            lbl_path = img_path.replace('images', 'labels', 1)
            lbl_path = os.path.splitext(lbl_path)[0] + ".txt"
            img_path = os.path.join(root_path, img_path.strip())
            lbl_path = os.path.join(root_path, lbl_path.strip())
            if idx == 0:
                height, width = cv2.imread(img_path).shape[:2]
            record["file_name"] = img_path
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
 
            objs = []
            with open(lbl_path, 'r') as annos:
                for anno in annos:
                    splits = anno.split(' ')
                    obj_cx = float(splits[1])
                    obj_cy = float(splits[2])
                    obj_w2 = float(splits[3]) / 2
                    obj_h2 = float(splits[4]) / 2
                    obj = {
                        "bbox": [
                            max(width * (obj_cx - obj_w2), 0),
                            max(height * (obj_cy - obj_h2), 0),
                            min(width * (obj_cx + obj_w2), width - 1),
                            min(height * (obj_cy + obj_h2), height - 1),
                        ],
                        "bbox_mode":
                        BoxMode.XYXY_ABS,
                        "category_id":
                        int(splits[0]),
                        "segmentation": [],
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts
 
for d in ["train", "val", "test"]:
    dataset_name = d + "_set"
    DatasetCatalog.register(dataset_name,
                            lambda d=d: load_dataset(data_info_file, d))
    MetadataCatalog.get(dataset_name).set(thing_classes=["car"])
 
dataset_metadata = MetadataCatalog.get("test_set")    
 
#CONFIG SETUP
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_set", )
cfg.TEST.EVAL_PERIOD = 1
cfg.DATALOADER.NUM_WORKERS = 0
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_weight_path
cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good
cfg.SOLVER.WARMUP_ITERS = 200
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (car). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
 
cfg.DATASETS.TEST = ("val_set", )
cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
cfg.SOLVER.GAMMA = 0.05
cfg.TEST.EVAL_PERIOD = 50
cfg.INPUT.MIN_SIZE_TEST = 1080
cfg.INPUT.MIN_SIZE_TRAIN = (920, 952, 984, 1016, 1048, 1080)
cfg.INPUT.MAX_SIZE_TEST = 1920
cfg.INPUT.MAX_SIZE_TRAIN = 1920
 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
cfg.MODEL.DEVICE = "cpu"
  
#PREDICTOR
predictor = DefaultPredictor(cfg)
 
#EVALUATOR AND INFERENCE
#https://detectron2.readthedocs.io/en/latest/modules/evaluation.html
#https://cocodataset.org/#detection-eval
#evaluator = COCOEvaluator("test_set", cfg, False, output_dir="./output")
evaluator = COCOEvaluator("test_set", tasks=["bbox"], distributed=False, output_dir="./output")

test_loader = build_detection_test_loader(cfg, "test_set")
 
results = inference_on_dataset(predictor.model, test_loader, evaluator)

with open("./output/result.txt", 'w') as writefile:
    writefile.write(json.dumps(results, indent=4))