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
from scipy.signal import butter, filtfilt
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
    # overlap_ratio = 0.5

    N = int((60 * 2 * Nyquist) / FResBPM)
    # N = _next_power_of_2(BVP.shape[0])
    # N = int(np.floor((Nyquist / FResBPM - 1) / (1 - overlap_ratio)))

    # Construct Periodogram
    F, Pxx = periodogram(BVP, window=np.hamming(len(BVP)), nfft=N, fs=FS)
    # F, Pxx = periodogram(BVP,  detrend=False, nfft=N, fs=FS)

    FMask = (F >= (LL_PR / 60)) & (F <= (UL_PR / 60))

    # Calculate predicted HR:
    FRange = F[FMask]
    PRange = Pxx[FMask]
    MaxInd = np.argmax(Pxx[FMask], axis=0)
    PR_F = FRange[MaxInd]
    PR = PR_F * 60

    return PR


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _calculate_fft_hr(ppg_signal, fs=25, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def filter_fxn(pre_HR, cur_HR, cap):
    gap = pre_HR - cur_HR
    if abs(gap) < cap:
        return cur_HR
    else:
        return pre_HR - np.sign(gap) * cap * log(abs(gap) + 1)


def predict_vitals(video_name, dir_path, data_set, filter):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = "../mtts_can.hdf5"
    batch_size = 10
    fs = 25

    sample_data_path = dir_path + video_name + ".mkv"

    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    dXsub_len = (dXsub.shape[0] // frame_depth) * frame_depth
    # print("Number of predicted frame:", dXsub_len)
    print(dXsub.shape)
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)
    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)

    # bandpass filter with range of [0.75, 2.5]
    # [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    [b_pulse, a_pulse] = butter(1, [(40 / 60) / fs * 2, (140 / 60) / fs * 2], btype='bandpass')
    pulse_pred = filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    # [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    [b_resp, a_resp] = butter(1, [0.080 / fs * 2, (40/60) / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))
    ppg_pred = pulse_pred + resp_pred
    plt.plot(ppg_pred, "r")
    plt.show()

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

    #######################################################################
    # Train dataset
    HR_predicted = np.ones(dXsub_len)
    ###################################################################################
    # # Train dataset =
    if data_set == "Train":
        with open("../../Phase1_data/Ground_truth/Physiology/" + video_name + ".txt") as f:
            contents = f.read()
            contents = contents.split(", ")
            start = 2
            end = start + dXsub_len
            # print('Number of ground truth frame:', end - start)
            HR_gt = [float(contents[start])]

            # # Cap of increase by logarithm increase
            # window_size = 1
            # pre_HR = prpsd(pulse_pred[0:100], fs, 40, 140)
            # cap = 3
            # for i in range(start + 1, end):
            #     if contents[i] == contents[i - 1]:
            #         window_size += 1
            #     else:
            #         if HR_predicted[0] != 1.0:
            #             pre_HR = HR_predicted[i - window_size - 3]
            #         HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
            #         HR_predicted[i - window_size - 2:i - 2] = filter_fxn(pre_HR,HR_pred_curr,cap)
            #         window_size = 1
            #     if i == end - 1:
            #         window_size += 1
            #         pre_HR = HR_predicted[i - window_size - 3]
            #         HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
            #         HR_predicted[i - window_size - 2:i - 2] = filter_fxn(pre_HR,HR_pred_curr,cap)
            #     HR_gt.append(float(contents[i][0:4]))
            # HR_gt = np.array(HR_gt)

            # # Constant window - Capping
            # pre_HR = prpsd(pulse_pred[0:100], fs, 40, 140)
            # window_size = 0
            # cap = 3
            # for i in range(start + 1, end):
            #     window_size += 1
            #     if window_size == 13:
            #         if HR_predicted[0] == 1.0:
            #             HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
            #         else:
            #             pre_HR = HR_predicted[i - window_size - 3]
            #             HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
            #         HR_predicted[i - window_size - 2:i - 2] = filter_fxn(pre_HR, HR_pred_curr, cap)
            #         window_size = 0
            #     if i == end - 1:
            #         window_size += 1
            #         pre_HR = HR_predicted[i - window_size - 3]
            #         HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
            #         HR_predicted[i - window_size - 2:i + 1] = filter_fxn(pre_HR, HR_pred_curr, cap)
            #     HR_gt.append(float(contents[i][0:4]))
            # HR_gt = np.array(HR_gt)

            # # Constant window - Previous Knowledge
            # pre_HR = prpsd(pulse_pred[0:100], fs, 40, 140)
            # window_size = 0
            # for i in range(start + 1, end):
            #     window_size += 1
            #     if window_size == 15:
            #         if HR_predicted[0] == 1.0:
            #             HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
            #         else:
            #             HR_pred_curr = prpsd(pulse_pred[i - window_size - 2 - 10:i - 2], fs, 40, 140)
            #         HR_predicted[i - window_size - 2:i - 2] = HR_pred_curr
            #         window_size = 0
            #     if i == end - 1:
            #         window_size += 1
            #         HR_pred_curr = prpsd(pulse_pred[i - window_size - 2 - 10:i - 2], fs, 40, 140)
            #         HR_predicted[i - window_size - 2:i + 1] = HR_pred_curr
            #     HR_gt.append(float(contents[i][0:4]))
            # HR_gt = np.array(HR_gt)

        # plt.plot(HR_predicted, "b", label="Prediction")
        # plt.plot(HR_gt, "r", label="Ground Truth")
        # plt.legend()
        # plt.show()

    ####################################################################################
    # # Validation Dataset
    # # Original
    elif data_set == "Valid":
        if not filter:
            with open("../../Phase2_data/validation_set_gt_release.txt") as f:
                contents = f.read()
                contents = contents.split(", ")
                indices = [i for i, s in enumerate(contents) if video_name + ".mkv" in s]
                start = indices[0] + 2
                end = start + dXsub_len
                window_size = 1
                HR_gt = [float(contents[start])]
                length = end - start
                for i in range(3, length + 2):
                    if contents[i + start] == contents[i + start - 1]:
                        window_size += 1
                    else:
                        HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 45, 150)
                        HR_predicted[i - window_size - 2:i - 2] = HR_pred_curr
                        window_size = 1
                    if i == length:
                        window_size += 1
                        HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 45, 150)
                        HR_predicted[i - window_size - 2:i + 1] = HR_pred_curr
                    try:
                        HR_gt.append(float(contents[i][0:4]))
                    except:
                        stop = i - 1
                        HR_predicted = HR_predicted[0:stop - 1]
                        break
                HR_gt = np.array(HR_gt)
        else:
            # With filtering
            with open("../../Phase2_data/validation_set_gt_release.txt") as f:
                contents = f.read()
                contents = contents.split(", ")
                indices = [i for i, s in enumerate(contents) if video_name + ".mkv" in s]
                start = indices[0] + 2
                end = start + dXsub_len
                window_size = 1
                HR_gt = [float(contents[start])]
                length = end - start
                # pre_HR = prpsd(pulse_pred[0:100], fs, 40, 140)
                pre_HR = 80
                for i in range(3, length + 2):
                    if contents[i + start] == contents[i + start - 1]:
                        window_size += 1
                    else:
                        if HR_predicted[0] == 1.0:
                            HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                        else:
                            pre_HR = HR_predicted[i - window_size - 3]
                            HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                        cap = 2 * window_size / fs
                        HR_predicted[i - window_size - 2:i - 2] = filter_fxn(pre_HR, HR_pred_curr, cap)
                        window_size = 1
                    if i == length + 1:
                        window_size += 1
                        cap = 2 * window_size / fs
                        pre_HR = HR_predicted[i - window_size - 3]
                        HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                        HR_predicted[i - window_size - 2:dXsub_len] = filter_fxn(pre_HR, HR_pred_curr, cap)
                    try:
                        HR_gt.append(float(contents[i][0:4]))
                    except:
                        stop = i - 1
                        cap = 2 * window_size / fs
                        pre_HR = HR_predicted[i - window_size - 3]
                        HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                        HR_predicted[i - window_size - 2:dXsub_len] = filter_fxn(pre_HR, HR_pred_curr, cap)
                        HR_predicted = HR_predicted[0:stop - 1]
                        break
                HR_gt = np.array(HR_gt)

    #############################################################################################
    # # Test dataset
    elif data_set == "Test":
        if not filter:
            # Original
            with open("../../Phase2_data/test_set_gt_release.txt") as f:
                contents = f.read()
                contents = contents.split(", ")
                indices = [i for i, s in enumerate(contents) if video_name + ".mkv" in s]
                start = indices[0] + 2
                end = start + dXsub_len
                window_size = 1
                HR_gt = [float(contents[start])]
                length = end - start
                for i in range(3, length + 2):
                    if contents[i + start] == contents[i + start - 1]:
                        window_size += 1
                    else:
                        HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 45, 150)
                        HR_predicted[i - window_size - 2:i - 2] = HR_pred_curr
                        window_size = 1
                    if i == length:
                        window_size += 1
                        HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 45, 150)
                        HR_predicted[i - window_size - 2:i + 1] = HR_pred_curr
                    try:
                        HR_gt.append(float(contents[i][0:4]))
                    except:
                        stop = i - 1
                        HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                        HR_predicted[i - window_size - 2:i - 2] = HR_pred_curr
                        HR_predicted = HR_predicted[0:stop - 1]
                        break
                HR_gt = np.array(HR_gt)
        else:
            # Filtering -- cap
            HR_predicted = np.ones(dXsub_len)
            with open("../../Phase2_data/test_set_gt_release.txt") as f:
                contents = f.read()
                contents = contents.split(", ")
                indices = [i for i, s in enumerate(contents) if video_name + ".mkv" in s]
                start = indices[0] + 2
                end = start + dXsub_len
                window_size = 1
                HR_gt = [float(contents[start])]
                length = end - start
                # pre_HR = prpsd(pulse_pred[0:100], fs, 40, 140)
                pre_HR = 80
                for i in range(3, length + 2):
                    if contents[i + start] == contents[i + start - 1]:
                        window_size += 1
                    else:
                        if HR_predicted[0] == 1.0:
                            HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                        else:
                            pre_HR = HR_predicted[i - window_size - 3]
                            HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                        cap = 2 * window_size / fs
                        HR_predicted[i - window_size - 2:i - 2] = filter_fxn(pre_HR, HR_pred_curr, cap)
                        window_size = 1
                    if i == length + 1:
                        window_size += 1
                        cap = 2 * window_size / fs
                        pre_HR = HR_predicted[i - window_size - 3]
                        HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                        HR_predicted[i - window_size - 2:dXsub_len] = filter_fxn(pre_HR, HR_pred_curr, cap)
                    try:
                        HR_gt.append(float(contents[i][0:4]))
                    except:
                        stop = i - 1
                        cap = 2 * window_size / fs
                        pre_HR = HR_predicted[i - window_size - 3]
                        HR_pred_curr = prpsd(pulse_pred[i - window_size - 2:i - 2], fs, 40, 140)
                        HR_predicted[i - window_size - 2:dXsub_len] = filter_fxn(pre_HR, HR_pred_curr, cap)
                        HR_predicted = HR_predicted[0:stop - 1]
                        break
                HR_gt = np.array(HR_gt)
    else:
        print("choose the correct datatype from: train, valid, test")

    # plt.plot(pulse_pred, "b", label="Prediction")
    # plt.plot(HR_predicted, "b", label="Prediction")
    # plt.plot(HR_gt, "r", label="Ground Truth")
    # plt.legend()
    # plt.show()

    cMAE = sum(abs(HR_predicted - HR_gt)) / dXsub_len
    cRMSE = np.sqrt(sum((abs(HR_predicted - HR_gt)) ** 2) / dXsub_len)
    cR, _ = pearsonr(HR_gt, HR_predicted)

    print("cMAE: ", cMAE)
    print("cRMSE: ", cRMSE)
    print("cR: ", cR)
    return cMAE, cRMSE, cR


if __name__ == "__main__":
    datatype = "Valid"
    if datatype != "Test":
        dir_path = "../../Phase1_data/Videos/" + datatype + "/"
    else:
        dir_path = "../../Phase2_data/Videos/" + datatype + "/"

    res = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    num_video = len(res)

    # results = [Parallel(n_jobs=6)(
    #     delayed(predict_vitals)(video[0:-4], dir_path, datatype, filter=True) for video in res)]
    # results = np.array(results)
    # MAE = results[0, :, 0]
    # RMSE = results[0, :, 1]
    # PC = results[0, :, 2]

    for i in range(num_video):
        print("Current Video:", res[i])
        video_name = res[i][0:-4]
        MAE, RMSE, PC = predict_vitals(video_name,dir_path, datatype, filter=True)

    # print("Average MAE:", mean(MAE))
    # print("Average RMSE:", mean(RMSE))
    # print("Average PC:", mean(PC))

    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 7), tight_layout=True)
    axs1[0].hist(MAE, bins=50)
    axs1[0].title.set_text("MAE")
    axs1[1].hist(RMSE, bins=50)
    axs1[1].title.set_text("RMSE")
    axs1[2].hist(RMSE, bins=50)
    axs1[2].title.set_text("PC")
    plt.show()

    # np.savetxt("../Error/MAE_ratio.txt", MAE, delimiter=",")
    # np.savetxt("../Error/RMSE_ratio.txt", RMSE, delimiter=",")
    # np.savetxt("../Error/PC_ratio.txt", RMSE, delimiter=",")
