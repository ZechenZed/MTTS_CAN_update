import os
import pandas as pd
import numpy as np
from statistics import mean, stdev
import cv2 as cv

if __name__ == "__main__":
    dir_path = "../../Phase1_data/Videos/train-001_of_002"
    res = []

    # df_MAE = pd.DataFrame()
    #
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    print(res[0])

    MAE = np.loadtxt("../MAE_001.txt")
    print("Total number of frame:", len(MAE))
    print("Mean of MAE", mean(MAE))
    print("Standard deviation:", stdev(MAE))
    src_error = []
    new_error = []
    for i in range(len(MAE)):
        if MAE[i] > 25:
            src_error.append(i)
        else:
            new_error.append(MAE[i])
    src_error = np.array(src_error)
    # print(src_error)
    print("Number of large error file:", len(src_error))
    print("Video with large error: ")
    for file in src_error:
        print(res[file])

    print("GETTING RID OF LARGE ERROR VIDEOS WITH HEAVY BODY MOVEMENT")
    print("Mean of MAE", mean(new_error))
    print("Standard deviation:", stdev(new_error))
