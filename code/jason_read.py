import cv2
import numpy as np
import glob
import os

img_array = []
dir_path = "../../PURE/08-01/08-01/"
res = []
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)
num_video = len(res)
print(res)
for filename in res:
    img = cv2.imread("../../PURE/08-01/08-01/"+filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.mkv', cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
