import prepare_data
import os
import sys
import yaml
from tqdm import tqdm
import detect
import subprocess

data_file = prepare_data.data_file

if not os.path.isfile(data_file):
    sys.exit(f"{data_file} not found! Have you run 'prepare_data.py'?")

with open(data_file) as file:
    data = yaml.safe_load(file)

path = data['path']
test_file_name = data['test']

test_file = os.path.join(path, test_file_name)
models_to_compare = [
    './runs/results/Paper - 3 Train - 2 Test Videos/Without Augmentation/train/yolov5s/weights/best.pt'
    './runs/results/Paper - 3 Train - 2 Test Videos/Without Augmentation/train/yolov8s/weights/best.pt'
]


# run a command and capture its output
result = subprocess.run(['echo', 'hello', 'world'], stdout=subprocess.PIPE)

# print the output
print(result.stdout.decode('utf-8'))


with open(test_file, "r") as file:
    lines = file.readlines()
    for img_path in tqdm(lines):
        img_path = os.path.join(path, img_path)
        for model in models_to_compare:
            opt.yolo_model = model
            opt.source = img_path
            detect.main(opt)
