import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
from scipy.signal import find_peaks
import os
import csv
import pandas as pd
from scipy.stats import pearsonr
import json
from math import log

# from numba import cuda, jit , cuda, uint32, f8, uint8

sys.path.append('../')
from model import Attention_mask, MTTS_CAN, TS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend

import numpy as np
from scipy.signal import periodogram
from joblib import Parallel, delayed
from statistics import mean


def prpsd(BVP, FS, LL_PR, UL_PR):
    """
    Estimates a pulse rate from a BVP signal.
    Inputs:
        BVP     - A BVP timeseries.
        FS      - The sample rate of the BVP time series (Hz/fps).
        LL_PR   - The lower limit for pulse rate (bpm).
        UL_PR   - The upper limit for pulse rate (bpm).
        PlotTF  - Boolean to turn plotting results on or off.
    Outputs:
        PR      - The estimated PR in BPM.
    """

    Nyquist = FS / 2
    FResBPM = 0.5  # resolution (bpm) of bins in power spectrum used to determine PR and SNR
    N = int((60 * 2 * Nyquist) / FResBPM)

    # Construct Periodogram
    F, Pxx = periodogram(BVP, window=np.hamming(len(BVP)), nfft=N, fs=FS)
    FMask = (F >= (LL_PR / 60)) & (F <= (UL_PR / 60))

    # Calculate predicted HR:
    FRange = F[FMask]
    PRange = Pxx[FMask]
    MaxInd = np.argmax(Pxx[FMask], axis=0)
    PR_F = FRange[MaxInd]
    PR = PR_F * 60

    return PR


def predict_vitals(video_name):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = "../mtts_can.hdf5"
    batch_size = 10
    fs = 25

    # sample_data_path = " ../../data/" + video_name + ".mkv"
    sample_data_path = " ../../Phase1_data/Videos/train/" + video_name + ".mkv"
    # sample_data_path = " ../../Phase2_data/Videos/Test/" + video_name + ".mkv"

    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    dXsub_len = (dXsub.shape[0] // frame_depth) * frame_depth
    # print("Number of predicted frame:", dXsub_len)
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)
    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    # bandpass filter with range of [0.7, 2.5]
    # [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    [b_pulse, a_pulse] = butter(1, [(40 / 60) / fs * 2, (140 / 60) / fs * 2], btype='bandpass')

    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    # resp_pred = yptest[1]
    # resp_pred = detrend(np.cumsum(resp_pred), 100)
    # [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    # resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    # # PURE dataset
    # f = open('../../PURE/08-01/08-01.json')
    # data = json.load(f)
    # HR_gt = []
    # for i in data['/FullPackage']:
    #     info_keys = i["Value"]
    #     HR_gt.append(info_keys["pulseRate"])
    #
    # f.close()
    # # Iterating through the json
    # # list
    # HR_gt = []
    # for i in data['/FullPackage']:
    #     info_keys = i["Value"]
    #     HR_gt.append(info_keys["pulseRate"])
    #
    # print(len(HR_gt))
    # HR_gt = HR_gt[0:dXsub_len]
    # # Customized test
    # for i in range(0, dXsub_len, 15):
    #     if i == 0:
    #         HR_pred_curr = prpsd(pulse_pred[i:i + 30], fs, 40, 140)
    #         HR_predicted[i:i + 15] = HR_pred_curr
    #     else:
    #         pre_HR = HR_predicted[i - 16]
    #         HR_pred_curr = prpsd(pulse_pred[i:i + 30], fs, 40, 140)
    #         HR_predicted[i:i + 15] = HR_pred_curr
    #
    # plt.plot(HR_predicted)
    # plt.title('Pulse Prediction')
    # plt.show()

    HR_predicted = np.ones(dXsub_len)
    with open("../../Phase1_data/Ground_truth/Physiology/" + video_name + ".txt") as f:
        contents = f.read()
        contents = contents.split(", ")
        start = 2
        end = start + dXsub_len
        # print('Number of ground truth frame:', end - start)
        HR_gt = [float(contents[start])]
        window_size = 1

        # # Cap of increase by linear gap
        # for i in range(start + 1, end):
        #     if contents[i] == contents[i - 1]:
        #         window_size += 1
        #     else:
        #         if HR_predicted[0] == 1.0:
        #             HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 80, 100)
        #         else:
        #             pre_HR = HR_predicted[i - window_size - 3]
        #             HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, max(60, pre_HR - 5),
        #                                  min(pre_HR + 5, 140))
        #         HR_predicted[i - window_size - 2:i - 2] = HR_pred_curr
        #         window_size = 1
        #     if i == end - 1:
        #         window_size += 1
        #         pre_HR = HR_predicted[i - window_size - 3]
        #         HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, max(60, pre_HR - 5),
        #                              min(pre_HR + 5, 140))
        #         HR_predicted[i - window_size - 2:i + 1] = HR_pred_curr
        #     HR_gt.append(float(contents[i][0:4]))
        # HR_gt = np.array(HR_gt)

        # # Cap of increase by logarithm increase
        pre_HR = 80
        cap = 3
        for i in range(start + 1, end):
            if contents[i] == contents[i - 1]:
                window_size += 1
            else:
                if HR_predicted[0] == 1.0:
                    HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                else:
                    pre_HR = HR_predicted[i - window_size - 3]
                    HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                gap = pre_HR - HR_pred_curr
                if gap == 0: gap = 0.001
                HR_predicted[i - window_size - 2:i - 2] = pre_HR - np.sign(gap) * (cap / log(cap)) * log(abs(gap))
                window_size = 1
            if i == end - 1:
                window_size += 1
                pre_HR = HR_predicted[i - window_size - 3]
                HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                gap = pre_HR - HR_pred_curr
                if gap == 0: gap = 0.001
                HR_predicted[i - window_size - 2:i + 1] = pre_HR - np.sign(gap) * (cap / log(cap)) * log(abs(gap))
            HR_gt.append(float(contents[i][0:4]))
        HR_gt = np.array(HR_gt)

        # # Constant window
        # pre_HR = prpsd(pulse_pred[0:int(dXsub_len / 4)], fs, 40, 140)
        # window_size = 0
        # for i in range(start + 1, end):
        #     window_size += 1
        #     if window_size == 13:
        #         if HR_predicted[0] == 1.0:
        #             HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
        #         else:
        #             pre_HR = HR_predicted[i - window_size - 3]
        #             HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
        #         gap = pre_HR - HR_pred_curr
        #         if gap == 0 : gap = 0.001
        #         HR_predicted[i - window_size - 2:i - 2] = pre_HR - np.sign(gap) * 3 * log(abs(gap))
        #         window_size = 0
        #     if i == end - 1:
        #         window_size += 1
        #         pre_HR = HR_predicted[i - window_size - 3]
        #         HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
        #         gap = pre_HR - HR_pred_curr
        #         if gap == 0 : gap = 0.001
        #
        #         HR_predicted[i - window_size - 2:i + 1] = pre_HR - np.sign(gap) * 3 * log(abs(gap))
        #     HR_gt.append(float(contents[i][0:4]))
        # HR_gt = np.array(HR_gt)

    # plt.plot(HR_predicted, "b", label="Prediction")
    # plt.plot(HR_gt, "r", label="Ground Truth")
    # plt.legend()
    # plt.show()

    # HR_predicted = np.ones(dXsub_len)
    # with open("../../Phase2_data/test_set_gt_release.txt") as f:
    #     contents = f.read()
    #     contents = contents.split(", ")
    #
    #     indices = [i for i, s in enumerate(contents) if video_name + ".mkv" in s]
    #     start = indices[0] + 2
    #     end = start + dXsub_len
    #
    #     # print('Number of ground truth frame:', end - start)
    #     window_size = 1
    #     HR_gt = [float(contents[start])]
    #     length = end - start
    #
    #     pre_HR = 80
    #     for i in range(3, length + 2):
    #         if contents[i + start] == contents[i + start - 1]:
    #             window_size += 1
    #         else:
    #             # print("check 1, window size:",window_size,"i:",i)
    #             if HR_predicted[0] == 1.0:
    #                 HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
    #             else:
    #                 pre_HR = HR_predicted[i - window_size - 3]
    #                 HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
    #             gap = pre_HR - HR_pred_curr
    #             if gap == 0: gap = 0.001
    #             HR_predicted[i - window_size - 2:i - 2] = pre_HR - np.sign(gap) * 3 * log(abs(gap))
    #             window_size = 1
    #         if i == length:
    #             window_size += 1
    #             pre_HR = HR_predicted[i - window_size - 3]
    #             HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
    #             gap = pre_HR - HR_pred_curr
    #             if gap == 0: gap = 0.001
    #             HR_predicted[i - window_size - 2:i + 1] = pre_HR - np.sign(gap) * 3 * log(abs(gap))
    #         try:
    #             HR_gt.append(float(contents[i][0:4]))
    #         except:
    #             stop = i - 1
    #             HR_predicted = HR_predicted[0:stop - 1]
    #             break
    #     HR_gt = np.array(HR_gt)

    # plt.plot(HR_predicted)
    # plt.title('Pulse Prediction')
    # plt.show()

    cMAE = sum(abs(HR_predicted - HR_gt)) / dXsub_len
    cRMSE = np.sqrt(sum((abs(HR_predicted - HR_gt)) ** 2) / dXsub_len)
    cR, _ = pearsonr(HR_gt, HR_predicted)

    print("MAE: ", cMAE)
    print("RMSE: ", cRMSE)
    print("cR", cR)
    return cMAE, cRMSE, cR


if __name__ == "__main__":
    # dir_path = "../../data"
    dir_path = "../../Phase1_data/Videos/train"
    # dir_path = "../../Phase2_data/Videos/Test"

    res = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    num_video = len(res)

    results = [Parallel(n_jobs=4)(delayed(predict_vitals)(video[0:-4]) for video in res)]
    results = np.array(results)
    MAE = results[0, :, 0]
    RMSE = results[0, :, 1]
    PC = results[0, :, 2]
    print("Average MAE:", mean(MAE))
    print("Average RMSE:", mean(RMSE))
    print("Average PC:", mean(PC))

    # for i in range(num_video):
    #     print("Current Video:", res[i])
    #     video_name = res[i][0:-4]
    #     MAE, RMSE, PC = predict_vitals(video_name)

    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 7), tight_layout=True)
    axs1[0].hist(MAE, bins=20)
    axs1[0].title.set_text("MAE")
    axs1[1].hist(RMSE, bins=20)
    axs1[1].title.set_text("RMSE")
    axs1[2].hist(RMSE, bins=20)
    axs1[2].title.set_text("PC")
    plt.show()

    # np.savetxt("../Error/MAE_ratio.txt", MAE, delimiter=",")
    # np.savetxt("../Error/RMSE_ratio.txt", RMSE, delimiter=",")
    # np.savetxt("../Error/PC_ratio.txt", RMSE, delimiter=",")
