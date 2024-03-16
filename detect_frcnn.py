import argparse
import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import cv2
from tqdm import tqdm

def create_detectron2_config():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_set", )
    cfg.TEST.EVAL_PERIOD = 1
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (car). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    ##### Modified by me ###########
    # cfg.MODEL.ROI_MASK_HEAD.NAME = None
    # cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = None
    # cfg.MODEL.SEM_SEG_HEAD.NAME = None
    cfg.DATASETS.TEST = ("val_set", )
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    cfg.SOLVER.GAMMA = 0.05
    cfg.TEST.EVAL_PERIOD = 50
    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.INPUT.MIN_SIZE_TRAIN = (920, 952, 984, 1016, 1048, 1080)
    cfg.INPUT.MAX_SIZE_TEST = 1920
    cfg.INPUT.MAX_SIZE_TRAIN = 1920
    return cfg

def detect(img_file_name: str, out_dir, predictor: DefaultPredictor):
    im = cv2.imread(img_file_name)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1])
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    f_name = os.path.join(out_dir, os.path.basename(img_file_name))
    cv2.imwrite(f_name, out.get_image()[:, :, ::-1])
    
def main(opt):
    run_name = os.path.basename(
        opt.model) if opt.name is None else opt.name
    out_dir = os.path.join(opt.project, run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    cfg = create_detectron2_config()
    cfg.MODEL.WEIGHTS = opt.model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opt.conf_thres
    predictor = DefaultPredictor(cfg)
    
    if os.path.isfile(opt.source):
        detect(opt.source, out_dir, predictor)
    elif os.path.isdir(opt.source):
        for filename in tqdm(os.listdir(opt.source)):
            file_path = os.path.join(opt.source, filename)
            if os.path.isfile(file_path):
                detect(file_path, out_dir, predictor)
    else:
        print(f"{opt.source} is neither a file nor a directory.")

def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        'This python script detect cars using a trained Faster R-CNN network.')
    parser.add_argument('--model',
                        type=str,
                        help='Faster R-CNN model to use for detection.',
                        required=True)
    parser.add_argument('--source', type=str, help='file/dir', required=True)
    parser.add_argument('--conf_thres',
                        type=float,
                        help='confidence threshold',
                        default=0.25)
    parser.add_argument(
        '--project',
        type=str,
        help="Project name. If omitted, is set to 'runs/detect'.",
        default='runs/detect')
    parser.add_argument('--name',
                        type=str,
                        help='Task name. If omitted, the model name is used.',
                        default=None)

    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
