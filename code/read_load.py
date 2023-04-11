import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # dir_path = "../Phase1_data/Videos/train-001_of_002"
    # res = []
    #
    # df_MAE = pd.DataFrame()
    #
    # for path in os.listdir(dir_path):
    #     if os.path.isfile(os.path.join(dir_path, path)):
    #         res.append(path)
    # print(res[0])
    MAE = np.loadtxt("../MAE_001.txt")
    print("Total number of frame:", len(MAE))
    src_error = []
    for i in range(len(MAE)):
        if MAE[i] > 25:
            src_error.append(i)
    print(len(src_error))
