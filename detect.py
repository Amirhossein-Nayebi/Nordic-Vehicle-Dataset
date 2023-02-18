import argparse
import os
import clearml
import yolov5.detect
import prepare_data
from ultralytics import YOLO as yolov8


def main(opt):
    yolo_model_name: str = opt.yolo_model if ".pt" in opt.yolo_model else opt.yolo_model + ".pt"
    conf: float = opt.conf_thres

    clearml.browser_login()

    project = opt.project
    run_name = os.path.basename(
        opt.yolo_model) if opt.name is None else opt.name

    if "yolov5" in yolo_model_name.lower():
        yolov5.detect.run(weights=yolo_model_name,
                          source=opt.source,
                          data=prepare_data.data_file,
                          imgsz=1920,
                          conf_thres=conf,
                          view_img=False,
                          save_txt=True,
                          project=project,
                          name=run_name)
    elif "yolov8" in yolo_model_name.lower():
        model = yolov8(yolo_model_name)
        model(source=opt.source,
              conf=conf,
              save=True,
              save_txt=True,
              project=project,
              name=run_name)


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        'This python script detect cars using a trained YOLO network.')
    parser.add_argument('--yolo_model',
                        type=str,
                        help='YOLO model to use for detection.',
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
