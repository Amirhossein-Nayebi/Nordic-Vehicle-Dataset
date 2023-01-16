import subprocess
import os
import cv2
import numpy as np
import time

file_dir = os.path.dirname(__file__)

ffd_dir = os.path.join(file_dir, '..', 'FFD/')
image_file = os.path.join(
    file_dir,
    "../ExtractFrames/Results/2022-12-02_Asjo_01.MP4/2_2-2_9/frame3111.png")
out_dir = file_dir

image = cv2.imread(image_file)

wsl_image_file = "/mnt/" + image_file.replace(':', '').replace('\\', '/')
wsl_ffd_dir = "/mnt/" + ffd_dir.replace(':', '').replace('\\', '/')
wsl_out_dir = "/mnt/" + out_dir.replace(':', '').replace('\\', '/')

max_keypoints = 10000
num_level = 3
contrast_threshold = 0.05
curvature_ratio = 10.
Time_cost = False

start = time.perf_counter()
result = subprocess.run([
    "wsl", wsl_ffd_dir + "/FFD", wsl_image_file, wsl_out_dir,
    str(num_level),
    str(max_keypoints),
    str(contrast_threshold),
    str(curvature_ratio),
    str(1 * Time_cost)
],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
end = time.perf_counter()

print(result.stdout.decode())
print(result.stderr.decode())
print("Elapsed time:", end - start, 'seconds')

resFile = os.path.join(out_dir, "FFD_" + os.path.basename(image_file) + ".txt")
with open(resFile, 'r') as file:
    lines = file.readlines()
    points_count = int(lines[1])
    keyPoints = np.zeros((points_count, 2))
    for line_number in range(0, points_count):
        line = lines[line_number + 2]
        splits = line.split(',')
        x = int(round(float(splits[0])))
        y = int(round(float(splits[1])))
        scale = float(splits[2])
        response = float(splits[3])
        keyPoints[line_number, :] = [x, y]
        cv2.drawMarker(image, (x, y), (0, 255, 0))

cv2.imshow("test", image)
cv2.waitKey(-1)