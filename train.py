import argparse
import clearml
import os
import sys
import yolov5.train
import prepare_data
from ultralytics import YOLO as yolov8
import yaml

available_models = [
    'yolov5n',
    'yolov5s',
    'yolov5m',
    'yolov5l',
    'yolov5x',
    'yolov8n',
    'yolov8s',
    'yolov8m',
    'yolov8l',
    'yolov8x',
]

hyp_file = './hyp.yaml'


def main(opt):
    if not os.path.isfile(prepare_data.data_file):
        sys.exit(
            f"'{prepare_data.data_file}' not found! Run 'python {prepare_data.__name__}.py --videos_dir <path/to/video/files> --data_dir <path/to/data/>'."
        )
    yolo_model_name: str = opt.yolo_model if ".pt" in opt.yolo_model else opt.yolo_model + ".pt"
    epochs: int = opt.epochs
    project_name: str = opt.project
    exp_name: str = opt.yolo_model if opt.name is None else opt.name
    batch_size = opt.batch
    clearml.browser_login()
    apply_augmentation = opt.aug

    if "yolov5" in yolo_model_name.lower():
        params = {
            'imgsz': 1920,
            'batch_size': batch_size,
            'data': prepare_data.data_file,
            'epochs': epochs,
            'weights': yolo_model_name,
            "project": project_name,
            "name": exp_name
        }
        if apply_augmentation:
            params['hyp'] = hyp_file
        yolov5.train.run(**params)
    elif "yolov8" in yolo_model_name.lower():
        try:
            model = yolov8(yolo_model_name)
            params = dict()
            if apply_augmentation:
                with open(hyp_file) as file:
                    params = yaml.safe_load(file)
                    # These parameters are not used in YOLOv8:
                    del params['obj_pw']
                    del params['anchor_t']
                    del params['iou_t']
                    del params['cls_pw']
                    del params['obj']
            params.update({
                'batch': batch_size,
                'imgsz': 1920,
                'data': prepare_data.data_file,
                'epochs': epochs,
                'model': yolo_model_name,
                'project': project_name,
                'name': exp_name
            })
            model.train(**params)
        except Exception as ex:
            print(ex)


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description='This python script fine tunes YOLO over new car images.')
    parser.add_argument('--epochs',
                        type=int,
                        help='total training epochs (default = 100)',
                        default=100)
    parser.add_argument(
        '--batch',
        type=int,
        help=
        'Training batch size. If you get out of memory error try to reduce the bach value. (default = 10)',
        default=10)

    parser.add_argument('--yolo_model',
                        type=str,
                        help='YOLO model to fine tune. Available models = ' +
                        str(available_models) + " (default = yolov5s)",
                        default='yolov5s')
    parser.add_argument(
        '--project',
        type=str,
        help="Project name. If omitted, is set to 'runs/train'.",
        default='runs/train')
    parser.add_argument('--name',
                        type=str,
                        help='Task name. If omitted, yolo model name is used.',
                        default=None)

    parser.add_argument('--aug',
                        action='store_true',
                        help='Apply online augmentation.')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
