import argparse
import os
import sys
import prepare_data
import yaml
from tqdm import tqdm


def main(opt):
    data_file = opt.data_file
    if not os.path.isfile(data_file):
        sys.exit(
            f"{data_file} not found! Have you run 'prepare_data.py'?"
        )

    with open(data_file) as file:
        data = yaml.safe_load(file)

    stats: dict = {}

    for data_type in ['train', 'val', 'test']:
        print(f"Analyzing {data_type} set...")
        data_path = data['path']
        list_file = os.path.join(data['path'], data[data_type])

        frames_count = 0
        cars_count = 0
        with open(list_file, "r") as file:
            lines = file.readlines()
            for img_path in tqdm(lines):
                img_full_path = os.path.join(data_path, img_path.strip())
                lbl_full_path = img_full_path.replace('images',
                                                      'labels').replace(
                                                          '.png', '.txt')
                if (os.path.isfile(lbl_full_path)):
                    frames_count += 1
                    with open(lbl_full_path, 'r') as file:
                        for line in file:
                            splits = line.strip().split(' ')
                            if len(splits) == 5:
                                cars_count += 1
        stats[data_type] = (frames_count, cars_count)
    print()
    for item in stats:
        print(item, "set")
        print(f"Number of frames: {stats[item][0]}")
        print(f"Number of cars: {stats[item][1]}")
        print()



def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description=
        "\nThis python script generates data statistics. You have to run 'prepare_data.py' first."
    )
    parser.add_argument('data_file',
                        type=str,
                        help='The data file.')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
