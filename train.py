import argparse
import clearml
import os
import sys
import yolov5.train
import prepare_data

yolov5_models = {
    'yolov5n',
    'yolov5s',
    'yolov5m',
    'yolov5l',
    'yolov5x',
}


def main(opt):

    dataFile = "./smart_plane.yaml"
    if not os.path.isfile(dataFile):
        sys.exit(
            f"'{dataFile}' not found! Run 'python {prepare_data.__name__}.py --videos_dir <path/to/video/files> --data_dir <path/to/data/>'."
        )

    # clearml.browser_login()

    yolov5.train.run(
        imgsz=1920,
        #    batch_size=16,
        data=dataFile,
        epochs=opt.epochs,
        weights=opt.yolo_model if ".pt" in opt.yolo_model else opt.yolo_model +
        ".pt",
        # cache=True,
        project=opt.project,
        name=opt.yolo_model if opt.name is None else opt.name)


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description='This python script fine tunes YOLO over new car images.')
    parser.add_argument('--epochs',
                        type=int,
                        help='total training epochs',
                        default=100)
    parser.add_argument('--yolo_model',
                        type=str,
                        help='YOLO model to fine tune. Available models = ' +
                        str(yolov5_models),
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
