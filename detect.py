import argparse
import yolov5.detect


def main(opt):
    # clearml.browser_login()
    yolov5.detect.run(weights=opt.yolo_model,
                  source=opt.source,
                  data="./smart_plane.yaml",
                  imgsz=1920,
                  conf_thres=opt.conf_thres,
                  view_img=True,
                  save_txt=True)


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        'This python script detect cars using a trained YOLO network.')
    parser.add_argument('--yolo_model',
                        type=str,
                        help='YOLO model to use for detection.',
                        required=True)
    parser.add_argument('--source',
                        type=str,
                        help='file/dir',
                        required=True)
    parser.add_argument('--conf_thres',
                        type=float,
                        help='confidence threshold',
                        default=0.25)

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
