import argparse
import clearml
import os
import sys
import yolov5.val


def main(opt):
    # clearml.browser_login()
    yolov5.val.run(
        weights=opt.yolo_model,
        project=opt.project,
        name=os.path.basename(opt.yolo_model) if opt.name is None else opt.name,
        task='test', # train, val, test, speed or study
        verbose=True,
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        imgsz=1920,
        single_cls=True,
        #    batch_size=16,
        data="./smart_plane.yaml")


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description='This python script validates a trained YOLO network.')
    parser.add_argument('--yolo_model',
                        type=str,
                        help='YOLO model to validate.',
                        required=True)
    parser.add_argument(
        '--project',
        type=str,
        help="Project name. If omitted, is set to 'runs/val'.",
        default='runs/val')
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
