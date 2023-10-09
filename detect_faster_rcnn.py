import os
import random
import torch, detectron2
import yaml
from tqdm import tqdm
import cv2
import numpy as np
 
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

model_weight_path = "/home/home/thesis/model/detectron2/faster_rcnn/model_final.pth"
image_path = "/home/home/thesis/code/faster_rcnn_demo/dataset_sample/images/2022-12-03 Nyland 01_stabilized-frame0012.png"

setup_logger()

MetadataCatalog.get("test_set").set(thing_classes=["car"])
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

#PREDICTION
im = cv2.imread(image_path)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata, scale=0.5,)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#SHOW IMAGE
cv2.imwrite('./test_image_output.png', out.get_image())
