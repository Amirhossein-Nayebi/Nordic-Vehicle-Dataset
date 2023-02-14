import argparse
import clearml
import os
import sys
import yolov5.train
import prepare_data
from ultralytics import YOLO as yolov8

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


def main(opt):
    if not os.path.isfile(prepare_data.data_file):
        sys.exit(
            f"'{prepare_data.data_file}' not found! Run 'python {prepare_data.__name__}.py --videos_dir <path/to/video/files> --data_dir <path/to/data/>'."
        )
    yolo_model_name: str = opt.yolo_model if ".pt" in opt.yolo_model else opt.yolo_model + ".pt"
    epochs: int = opt.epochs
    project_name: str = opt.project
    exp_name: str = opt.yolo_model if opt.name is None else opt.name

    clearml.browser_login()

    if "yolov5" in yolo_model_name.lower():
        yolov5.train.run(
            imgsz=1920,
            #    batch_size=16,
            data=prepare_data.data_file,
            epochs=epochs,
            weights=yolo_model_name,
            # cache=True,
            project=project_name,
            name=exp_name)
    elif "yolov8" in yolo_model_name.lower():
        try:
            model = yolov8(yolo_model_name)
            model.train(imgsz=1920,
                        data=prepare_data.data_file,
                        epochs=epochs,
                        model=yolo_model_name,
                        project=project_name,
                        name=exp_name)
        except Exception as ex:
            print(ex)

def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description='This python script fine tunes YOLO over new car images.')
    parser.add_argument('--epochs',
                        type=int,
                        help='total training epochs (default = 100)',
                        default=100)
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

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
