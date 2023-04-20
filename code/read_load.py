import os
import pandas as pd
import numpy as np
from statistics import mean, stdev
import cv2 as cv
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # dir_path = "../../Phase1_data/Videos/train-001_of_002"
    # res = []

    # df_MAE = pd.DataFrame()
    #
    # for path in os.listdir(dir_path):
    #     if os.path.isfile(os.path.join(dir_path, path)):
    #         res.append(path)
    # print(res[0])

    MAE = np.loadtxt("../Error/MAE_002_fc_opt.txt")
    RMSE = np.loadtxt("../Error/RMSE_002_fc_opt.txt")
    PC = np.loadtxt("../Error/PC_002_fc_opt.txt")

    # print("Total number of frame:", len(MAE))
    # print("Mean of MAE", mean(MAE))
    # print("Standard deviation:", stdev(MAE))
    # src_error = []
    # new_error = []
    # for i in range(len(MAE)):
    #     if MAE[i] > 25:
    #         src_error.append(i)
    #     else:
    #         new_error.append(MAE[i])
    # src_error = np.array(src_error)
    # # print(src_error)
    # print("Number of large error file:", len(src_error))
    # print("Video with large error: ")
    # for file in src_error:
    #     print(res[file])
    #
    # print("GETTING RID OF LARGE ERROR VIDEOS WITH HEAVY BODY MOVEMENT")
    # print("Mean of MAE", mean(new_error))
    # print("Standard deviation:", stdev(new_error))
    fig1, axs1 = plt.subplots(2, 1, figsize=(8, 4.5), tight_layout=True)
    axs1[0].hist(MAE, bins=30)
    axs1[0].title.set_text("MAE")
    axs1[1].hist(RMSE, bins=30)
    axs1[1].title.set_text("RMSE")
    # axs1[2].hist(PC, bins=30)
    # axs1[2].title.set_text("PC")
    plt.show()

    # fig = plt.figure()
    # ax1 = fig.add_subplot(311)
    # ax2 = fig.add_subplot(312)
    # # ax3 = fig.add_subplot(313)
    # ax1.hist(MAE, bins=30)
    # ax1.title.set_text('MAE')
    # ax2.title.set_text('Second Plot')
    # ax2.hist(RMSE, bins=30)
    # # ax3.title.set_text('Third Plot')
    # # ax3.hist(PC, bins=30)
    #
    # plt.show()