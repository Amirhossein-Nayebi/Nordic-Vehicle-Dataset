import argparse
import clearml
import os
import sys
import yolov5.val
from ultralytics import YOLO as yolov8
import prepare_data


def main(opt):
    if not os.path.isfile(prepare_data.data_file):
        sys.exit(
            f"'{prepare_data.data_file}' not found! Run 'python {prepare_data.__name__}.py --videos_dir <path/to/video/files> --data_dir <path/to/data/>'."
        )

    yolo_model_name: str = opt.yolo_model if ".pt" in opt.yolo_model else opt.yolo_model + ".pt"

    clearml.browser_login()

    project = opt.project
    run_name = os.path.basename(
        opt.yolo_model) if opt.name is None else opt.name
    speed_file = run_name + "_speed.txt"
    imgsz = 1920
    data_split = 'test'  # train, val, test, speed or study
    verbose = True
    save_txt = False  # save results to *.txt
    save_hybrid = False  # save label+prediction hybrid results to *.txt
    save_conf = False  # save confidences in --save-txt labels
    save_json = False  # save a COCO-JSON results file
    single_cls = True

    if "yolov5" in yolo_model_name.lower():
        res = yolov5.val.run(
            weights=opt.yolo_model,
            project=project,
            name=run_name,
            task=data_split,  # train, val, test, speed or study
            verbose=verbose,
            save_txt=save_txt,  # save results to *.txt
            save_hybrid=
            save_hybrid,  # save label+prediction hybrid results to *.txt
            save_conf=save_conf,  # save confidences in --save-txt labels
            save_json=save_json,  # save a COCO-JSON results file
            imgsz=imgsz,
            single_cls=single_cls,
            #    batch_size=16,
            data=prepare_data.data_file)
        t = res[3]
        shape = (3, imgsz, imgsz)
        with open(speed_file, "w") as file:
            file.write(
                f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}'
                % t)

    elif "yolov8" in yolo_model_name.lower():
        model = yolov8(yolo_model_name)  # load a custom model
        # Validate the model
        metrics = model.val(
            data=prepare_data.data_file,
            split=data_split,
            project=project,
            name=run_name,
            verbose=verbose,
            save_txt=save_txt,  # save results to *.txt
            save_hybrid=
            save_hybrid,  # save label+prediction hybrid results to *.txt
            save_conf=save_conf,  # save confidences in --save-txt labels
            save_json=save_json,  # save a COCO-JSON results file
            imgsz=imgsz,
            single_cls=single_cls,
        )


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description='This python script validates a trained YOLO network.')
    parser.add_argument('--yolo_model',
                        type=str,
                        help='YOLO model to validate.',
                        required=True)
    parser.add_argument('--project',
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
